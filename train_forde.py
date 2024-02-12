import json
import logging
import os
from bisect import bisect
from itertools import chain
from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver
from scipy.stats import entropy
from checkpointer import Checkpointer
import models
from datasets import get_corrupt_data_loader, get_data_loader
# from trainer import ParVITrainer
import optax
import jax.example_libraries.optimizers as optimizers
from pmap_optimizers import nesterov_weight_decay
import tree

class SetID(RunObserver):
    priority = 50  # very high priority

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        return f"{config['model_name']}_{config['seed']}_{config['dataset']}_{config['name']}"


EXPERIMENT = 'experiments'
BASE_DIR = EXPERIMENT
ex = Experiment(EXPERIMENT)
ex.observers.append(SetID())
ex.observers.append(FileStorageObserver(BASE_DIR))


@ex.config
def my_config():
    ece_bins = 15
    seed = 1  # Random seed
    name = 'name'  # Unique name for the folder of the experiment
    model_name = 'StoResNet18'  # Choose with model to train
    batch_size = 128  # Batch size
    test_batch_size = 512
    n_members = 2  # Number of components in the posterior
    # Options of the deterministic weights for the SGD
    weight_decay = 5e-4
    init_lr = 0.1
    # Universal options for the SGD
    sgd_params = {
        'momentum': 0.9,
        'nesterov': True
    }
    num_epochs = 300  # Number of training epoch
    validation = True  # Whether of not to use a validation set
    validation_fraction = 0.1  # Size of the validation set
    save_freq = 301  # Frequency of saving checkpoint
    num_test_sample = 1  # Number of samples drawn from each component during testing
    logging_freq = 1  # Logging frequency
    device = 'cuda'
    lr_ratio = 0.01  # For annealing the learning rate of the deterministic weights
    # First value chooses which epoch to start decreasing the learning rate and the second value chooses which epoch to stop. See the schedule function for more information.
    milestones = (0.5, 0.9)
    augment_data = True
    # if not torch.cuda.is_available():
    #     device = 'cpu'
    dataset = 'cifar100'  # Dataset of the experiment
    if dataset == 'cifar100' or dataset == 'vgg_cifar100':
        num_classes = 100
        input_size = (32, 32, 3)
    elif dataset == 'cifar10' or dataset == 'vgg_cifar10' or dataset == 'fmnist':
        num_classes = 10
        input_size = (32, 32, 3)
    elif dataset == 'tinyimagenet':
        num_classes = 200
        input_size = (64, 64, 3)

    num_train_workers = 8
    num_test_workers = 2
    data_norm_stat = None
    num_start_epochs = 5
    alpha = {
        'start': 100.0, 'end': 1.0, 'n_epochs': 80
    }
    repulsive_type = 'jacobian'
    basis_coeffs_path = ""
    eps = 1e-12
    mean_over_batch = True
    empcov_path = ""
    share_bias = False
    initial_conv_config = { "kernel_shape": 3, "output_channels": 64, "stride": 1, "padding": "SAME"}
#     if model_name.startswith('WideResNet'):
#         first_conv_name = 'init_conv'
#     elif model_name.startswith('ResNet') or model_name.startswith('PreActResNet'):
    first_conv_name = 'initial_conv'
    label_smoothing = 0.0
    log_softmax_grad = False

@ex.capture
def get_alpha(alpha, num_epochs):
    def get_scale(epoch, start, end, n_epochs):
        value = start + epoch * (end-start)/n_epochs
        return value if epoch < n_epochs else end
    values = jnp.array([
        get_scale(i, **alpha) for i in range(num_epochs)
    ], dtype=jnp.float32)
    return values

class LrScheduler():
    def __init__(self, init_value, num_epochs, milestones, lr_ratio, num_start_epochs):
        self.init_value = init_value
        self.num_epochs = num_epochs
        self.milestones = milestones
        self.lr_ratio = lr_ratio
        self.num_start_epochs = num_start_epochs
        self.lrs = jnp.array([self.__lr(i) for i in range(num_epochs)], dtype=jnp.float32)
    
    def __call__(self, i):
        return self.lrs[i]

    def __lr(self, epoch):
        if epoch < self.num_start_epochs:
            return self.init_value * (1.0 - self.lr_ratio)/self.num_start_epochs * epoch + self.lr_ratio
        t = epoch / self.num_epochs
        m1, m2 = self.milestones
        if t <= m1:
            factor = 1.0
        elif t <= m2:
            factor = 1.0 - (1.0 - self.lr_ratio) * (t - m1) / (m2 - m1)
        else:
            factor = self.lr_ratio
        return self.init_value * factor

@ex.capture
def get_model(model_name, num_classes, input_size, keys, initial_conv_config):
    model_fn = getattr(models, model_name)
    def _forward(x, is_training):
        model = model_fn(num_classes, initial_conv_config=initial_conv_config)
        return model(x, is_training)
    forward = hk.transform_with_state(_forward)
    parallel_init_fn = jax.vmap(forward.init, (0, None, None), 0)
    # parallel_apply_fn = jax.vmap(forward.apply, (0, 0, None, None, None), 0)

  
    params, state = parallel_init_fn(keys, jnp.ones((1, *input_size)), True)

    return params, state

@ex.capture
def get_optimizer(init_lr, milestones, num_epochs, lr_ratio, num_start_epochs, sgd_params):
    scheduler = LrScheduler(init_lr, num_epochs, milestones, lr_ratio, num_start_epochs)
    opt_init, opt_update, get_params, get_velocity = nesterov_weight_decay(mass=sgd_params['momentum'], weight_decay=0.0)
    return opt_init, opt_update, get_params, get_velocity, scheduler

def l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)

@ex.capture
def get_dataloader(batch_size, test_batch_size, validation, validation_fraction, dataset, augment_data, num_train_workers, num_test_workers):
    return get_data_loader(dataset, train_bs=batch_size, test_bs=test_batch_size, validation=validation, validation_fraction=validation_fraction,
                           augment = augment_data, num_train_workers = num_train_workers, num_test_workers = num_test_workers)

@ex.capture
def get_corruptdataloader(intensity, test_batch_size, dataset, num_test_workers):
    return get_corrupt_data_loader(dataset, intensity, batch_size=test_batch_size, root_dir='data/', num_workers=num_test_workers)

@ex.capture
def get_logger(_run, _log):
    fh=logging.FileHandler(os.path.join(BASE_DIR, _run._id, 'train.log'))
    formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    _log.addHandler(fh)
    _log.setLevel(logging.INFO)
    return _log

def ece(softmax_logits, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(softmax_logits, -1), np.argmax(softmax_logits, -1)
    accuracies = predictions == labels

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)

    return ece

@ex.capture
def test_ensemble(model_fn, params, state, dataloader, ece_bins, n_members):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    devices = jax.local_devices()
    n_devices = len(devices)
    @partial(jax.pmap, axis_name='batch')
    def eval_batch(bx, by):
        logits, _ = model_fn(params, state, None, bx, False)
        bnll = -optax.softmax_cross_entropy_with_integer_labels(
            logits, by[None, :]
        )
        if n_members > 1:
            bnll = jax.scipy.special.logsumexp(bnll, axis=0) - jnp.log(jnp.array(bnll.shape[0], dtype=jnp.float32))
        else:
            bnll = jax.scipy.special.logsumexp(bnll, axis=0)
        prob = jax.nn.softmax(logits, axis=2)
        vote = prob.mean(axis=0)
        top3 = jax.lax.top_k(vote, k=3)[1]
        return bnll, prob, vote, top3

    for bx, by in dataloader:
        bx = jnp.array(bx.permute(0, 2, 3, 1).numpy())
        by = jnp.array(by.numpy())
        bx = jax.device_put_sharded(jnp.split(bx, n_devices, axis=0), devices)
        by = jax.device_put_sharded(jnp.split(by, n_devices, axis=0), devices)
        bnll, prob, vote, top3 = eval_batch(bx, by)
        bnll = jnp.concatenate(bnll, axis=0)
        prob = jnp.concatenate(prob, axis=1)
        vote = jnp.concatenate(vote, axis=0)
        top3 = jnp.concatenate(top3, axis=0)
        by = jnp.concatenate(by, axis=0)
        tnll -= bnll.sum()
        y_prob_all.append(prob)
        y_prob.append(vote)
        y_true.append(by)
        y_miss = top3[:, 0] != by
        if y_miss.sum() > 0:
            nll_miss -= bnll[y_miss].sum()
        for k in range(3):
            acc[k] += (top3[:, k] == by).sum()
    nll_miss /= len(dataloader.dataset) - acc[0]
    tnll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
    acc = jnp.cumsum(jnp.array(acc))
    y_prob = jnp.concatenate(y_prob, axis=0)
    y_true = jnp.concatenate(y_true, axis=0)
    y_prob_all = jnp.concatenate(y_prob_all, axis=1)
    total_entropy = jax.scipy.special.entr(y_prob).sum(1)
    aleatoric = jax.scipy.special.entr(y_prob_all).sum(axis=2).mean(axis=0)
    epistemic = total_entropy - aleatoric
    ece_val = ece(y_prob, y_true, ece_bins)
    result = {
        'nll': float(tnll),
        'nll_miss': float(nll_miss),
        'ece': float(ece_val),
        'predictive_entropy': {
            'total': (float(total_entropy.mean()), float(total_entropy.std())),
            'aleatoric': (float(aleatoric.mean()), float(aleatoric.std())),
            'epistemic': (float(epistemic.mean()), float(epistemic.std()))
        },
        **{
            f"top-{k}": float(a) for k, a in enumerate(acc, 1)
        }
    }
    return result

def select_first(x):
    for i in range(1, len(x)):
        assert jnp.allclose(x[0], x[i])
    return x[0]

@ex.automain
def main(_run, model_name, empcov_path, initial_conv_config, weight_decay, first_conv_name, num_classes, validation, num_epochs, batch_size, dataset, repulsive_type, seed, n_members, eps, input_size, basis_coeffs_path, mean_over_batch, label_smoothing, log_softmax_grad):
    logger=get_logger()
    if validation:
        train_loader, valid_loader, test_loader = get_dataloader()
        logger.info(
            f"Train size: {len(train_loader.dataset)}, validation size: {len(valid_loader.dataset)}, test size: {len(test_loader.dataset)}")
    else:
        train_loader, test_loader= get_dataloader()
        logger.info(
            f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    n_batch= len(train_loader)
    devices = jax.local_devices()
    n_devices = len(devices)
    rng = jax.random.PRNGKey(seed)
    key, *subkeys = jax.random.split(rng, n_members+1)
    subkeys = jnp.vstack(subkeys)
    params, state = get_model(keys=subkeys)
    
    model_fn = getattr(models, model_name)
    def _forward(x, is_training):
        model = model_fn(num_classes, bn_config={'cross_replica_axis': 'batch'}, initial_conv_config=initial_conv_config)
        return model(x, is_training)
    apply_fn = hk.transform_with_state(_forward).apply
    
    opt_init, opt_update, get_params, get_velocity, scheduler = get_optimizer()
    velocity = jax.tree_util.tree_map(opt_init, params)
    
    params = jax.device_put_replicated(params, devices)
    state = jax.device_put_replicated(state, devices)
    velocity = jax.device_put_replicated(velocity, devices)
    
    alphas = get_alpha()
    alphas = jax.device_put_replicated(alphas, devices)
    
    if repulsive_type == 'jacobian':
        matrix_basis = np.load(basis_coeffs_path)
        loaded_basis = jnp.array(matrix_basis['basis'])
        loaded_coeffs = jnp.array(matrix_basis['coeffs'])
    else:
        loaded_basis = loaded_coeffs = None
        
    step_sizes = jax.device_put_replicated(scheduler.lrs, devices)

    if empcov_path != "":
        empcov = np.load(empcov_path)
    else:
        empcov = None

    @partial(jax.pmap, axis_name='batch')
    def train_step(epoch, params, state, velocity, bx, by, alphas, step_size):
        batch_size, height, width, channel = bx.shape
        alpha = alphas[epoch]
        
        def forward(params, state, bx, by):
            logits, vjpfun, new_state = jax.vjp(lambda inputs: apply_fn(params, state, None, inputs, True), bx, has_aux=True)
            labels = jax.nn.one_hot(by, num_classes=logits.shape[-1], dtype=jnp.float32)
            logits_grad = jax.grad(lambda logits: jnp.sum(logits * labels))(logits)
            jacobian = vjpfun(logits_grad)[0]
            return (logits, jacobian), new_state
        
        def get_repulsive_term(jacobian):
            # jacobian: [n_members, n_devices, batch_size, *input_shape]
            jacobian       = jnp.reshape(jacobian, (n_members, n_devices * batch_size, height*width*channel)) @ loaded_basis
            jacobian       = jacobian / jnp.sqrt(jnp.sum(jnp.square(jacobian), axis=2, keepdims=True) + eps)
            jacobian       = jacobian * loaded_coeffs
            sqdist         = jax.vmap(jax.vmap(lambda x, y: jnp.sum(jnp.square(x-y), axis=1), (0, None), 0), (1, 1), 2)(jacobian, jax.lax.stop_gradient(jacobian))
            median         = jnp.median(jax.lax.stop_gradient(sqdist), 0)
            bandwidth      = median / jnp.log(jacobian.shape[0]) + 1e-12
            kernel_matrix  = jax.scipy.special.logsumexp(-sqdist/bandwidth, axis=(1, 2)) - jnp.log(jacobian.shape[1])
            repulsion_term = jnp.sum(kernel_matrix, axis=0) / batch_size
            median         = jnp.mean(median)
            return repulsion_term, median
            
        def calculate_gradients(params, state):
            (logits, jacobian), vjpfun, new_state = jax.vjp(lambda params: jax.vmap(forward, (0, 0, None, None), 0)(params, state, bx, by), params, has_aux=True)
            labels = jax.nn.one_hot(by, num_classes=logits.shape[-1], axis=-1, dtype=jnp.float32)
            if label_smoothing > 0.0:
                labels = (1.0 - label_smoothing) * labels + label_smoothing * jnp.ones_like(labels)/logits.shape[-1]
            cross_ent_loss, logits_grad = jax.value_and_grad(lambda logits: -(jax.nn.log_softmax(logits, axis=-1) * labels).sum(-1).mean(1).sum())(logits)
            jacobian = jax.lax.all_gather(jacobian, axis_name='batch', axis=1, tiled=False) # [n_members, n_devices, batch_size, *input_shape]
            (repulsion_term, median), jacobian_grad = jax.value_and_grad(get_repulsive_term, has_aux=True)(jacobian)
            jacobian_grad = jacobian_grad[:, jax.lax.axis_index('batch')]
            params_grad = vjpfun((logits_grad, alpha*jacobian_grad))[0]
            return params_grad, cross_ent_loss/n_members, repulsion_term/n_members/n_devices, alpha, median, new_state
        
        grads, cross_ent_loss, repulsion_term, repulsion_weight, median, new_state = calculate_gradients(params, state)
        grads         = jax.lax.pmean(grads, axis_name='batch')
        grads         = jax.tree_util.tree_map(lambda g, p: g + weight_decay * p, grads, params)
        new_opt_state = jax.tree_util.tree_map(lambda g, p, v: opt_update(step_size[epoch], g, p, v), grads, params, velocity)
        new_params    = jax.tree_util.tree_map(get_params, new_opt_state, is_leaf=lambda o: isinstance(o, tuple) and len(o)==2)
        new_velocity  = jax.tree_util.tree_map(get_velocity, new_opt_state, is_leaf=lambda o: isinstance(o, tuple) and len(o)==2)
        return new_state, new_params, new_velocity, jax.lax.pmean(cross_ent_loss, axis_name='batch'), jax.lax.pmean(repulsion_term, axis_name='batch'), jax.lax.pmean(repulsion_weight, axis_name='batch'), jax.lax.pmean(median, axis_name='batch')

    # trainer = ParVITrainer(model, optimizer, n_members, torch.nn.functional.nll_loss)
#     os.makedirs(os.path.join(BASE_DIR, _run._id, 'mask'), exist_ok=True)
    for i in range(num_epochs):
        total_loss = 0
        n_count = 0
        for bx, by in train_loader:
            bx = jnp.array(bx.permute(0, 2, 3, 1).numpy())
            by = jnp.array(by.numpy())
            bx = jax.device_put_sharded(jnp.split(bx, n_devices, axis=0), devices)
            by = jax.device_put_sharded(jnp.split(by, n_devices, axis=0), devices)
            epochs = jax.device_put_replicated(jnp.array(i), devices)
            state, params, velocity, nll_loss, repulsion_term, alpha, median = train_step(epochs, params, state, velocity, bx, by, alphas, step_sizes)
            nll_loss = jnp.mean(nll_loss)
            repulsion_term = jnp.mean(repulsion_term)
            alpha = jnp.mean(alpha)
            median = jnp.mean(median)
            total_loss += nll_loss
            n_count += 1
            logger.info(f"Epoch {i}: neg_log_like {nll_loss:.4f}, repulsion term {repulsion_term:.1e}, median {median:.4f}, alpha {alpha:.1f}, lr {scheduler(i).item():.4f}")
        ex.log_scalar("nll.train", total_loss / n_count, i)
    checkpointer = Checkpointer(os.path.join(BASE_DIR, _run._id, f'checkpoint.pkl'))
    checkpointer.save({'params': jax.tree_util.tree_map(select_first, params), 'state': jax.tree_util.tree_map(select_first, state)}) #, 'basis': loaded_basis, 'matrix': loaded_matrix})
    logger.info('Save checkpoint')
    param_state = checkpointer.load()
    params = param_state['params']
    state = param_state['state']
    parallel_apply_fn = jax.vmap(apply_fn, (0, 0, None, None, None), 0)
    test_result = test_ensemble(parallel_apply_fn, params, state, test_loader)
    os.makedirs(os.path.join(BASE_DIR, _run._id, dataset), exist_ok=True)
    with open(os.path.join(BASE_DIR, _run._id, dataset, 'test_result.json'), 'w') as out:
        json.dump(test_result, out)
    if validation:
        valid_result = test_ensemble(parallel_apply_fn, params, state, valid_loader)
        with open(os.path.join(BASE_DIR, _run._id, dataset, 'valid_result.json'), 'w') as out:
            json.dump(valid_result, out)
    for i in range(5):
        dataloader = get_corruptdataloader(intensity=i)
        result = test_ensemble(parallel_apply_fn, params, state, dataloader)
        os.makedirs(os.path.join(BASE_DIR, _run._id, dataset, str(i)), exist_ok=True)
        with open(os.path.join(BASE_DIR, _run._id, dataset, str(i), 'result.json'), 'w') as out:
            json.dump(result, out)


