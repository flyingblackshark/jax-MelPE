import os
import time
import logging
import tqdm
import flax
import jax
import optax
from flax import linen as nn
import jax.numpy as jnp
import orbax.checkpoint
from functools import partial
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
from flax.training import orbax_utils
from .model import MelPETransformer
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import MIR1K
import orbax
import jax.dlpack as jdp
import torch.utils.dlpack as tdp
import numpy as np

PRNGKey = jnp.ndarray
SAMPLE_RATE = 16000

N_CLASS = 360

N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
CONST = 1997.3794084376191



def create_train_state(rng,hp,trainloader): 
    r"""Create the training state given a model class. """ 
    model = MelPETransformer()
    
    exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps,decay_rate=hp.train.lr_decay)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.lion(learning_rate=exponential_decay_scheduler, b1=hp.train.betas[0],b2=hp.train.betas[1]))
        
    data = next(iter(trainloader))
    init_rngs = {'params': rng}
    mel = data['mel']
    mel = jnp.asarray(mel)
    variables = model.init(init_rngs, mel)
    state = TrainState.create(apply_fn=model.apply, tx=tx, params=variables['params'])
    
    return state
def bce(logits,labels):
    log_p = jnp.log(logits)
    log_not_p = jnp.log(1. -logits)
    return -labels * log_p - (1. - labels) * log_not_p
@partial(jax.pmap, axis_name='num_devices')
def combine_step(generator_state: TrainState, mel : jnp.ndarray  , pit : jnp.ndarray):
    
    def loss_fn(params):
        pit_pred = generator_state.apply_fn({'params': params},mel)
        loss = bce(pit_pred, pit).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(generator_state.params)

    # Average across the devices.
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    new_generator_state = generator_state.apply_gradients(grads=grads)

    return new_generator_state,loss
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm

def to_local_average_cents(salience, center=None, thred=0.03):
    cents_mapping = (20 * np.arange(N_CLASS) + CONST)
    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :], None, thred) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")

from collections import defaultdict


def validate(generator,validation_dataset):
        hop_length = 160
        generator = flax.jax_utils.unreplicate(generator)

        metrics = defaultdict(list)
        for data in validation_dataset:
            mel = data['mel'].numpy()
            pitch_label = data['pitch'].numpy()
            mel = jnp.asarray(mel)
            pitch_label = jnp.asarray(pitch_label)
            mel = jnp.expand_dims(mel,0)
            val_time = time.perf_counter() 
            pitch_pred = jax.jit(generator.apply_fn)({'params': generator.params}, mel).squeeze(0)
            val_time_cost = time.perf_counter() - val_time
            print("validate time cost:" + str(val_time_cost))
            pitch_pred = pitch_pred[ : pitch_label.shape[0]]
          
            loss = bce(pitch_pred, pitch_label).mean()
            metrics['loss'].append(loss)
            
            pitch_pred = np.asarray(pitch_pred)
            pitch_label = np.asarray(pitch_label)
            
            pitch_th = 0.0
            cents_pred = to_local_average_cents(pitch_pred, None, pitch_th)
            cents_label = to_local_average_cents(pitch_label, None, pitch_th)

            freq_pred = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
            freq = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents_label])

            time_slice = np.array([i*hop_length*1000/SAMPLE_RATE for i in range(len(cents_label))])
            ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freq, time_slice, freq_pred)

            rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
            rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
            oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
            vfa = voicing_false_alarm(ref_v, est_v)
            vr = voicing_recall(ref_v, est_v)
            metrics['RPA'].append(rpa)
            metrics['RCA'].append(rca)
            metrics['OA'].append(oa)
            metrics['VFA'].append(vfa)
            metrics['VR'].append(vr)
            # if rpa < 0.9:
            print(data['file'], ':\t', rpa, '\t', oa)
        return metrics
def train(args,chkpt_path, hp):
    hop_length = 160
    key = jax.random.PRNGKey(seed=hp.train.seed)
    
    init_epoch = 1
    step = 0
    pth_dir = os.path.join(hp.log.pth_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    os.makedirs(pth_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    train_dataset = MIR1K('Hybrid', hop_length, ['train'], whole_audio=False, use_aug=True)
    validation_dataset = MIR1K('Hybrid', hop_length, ['test'], whole_audio=True, use_aug=False)
    trainloader = DataLoader(train_dataset, hp.train.batch_size, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=8)

    new_train_state = create_train_state(key,hp,trainloader)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=10, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/mel2f0', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'new_train_state': new_train_state}
        step = checkpoint_manager.latest_step()  # step = 4
        new_train_state=checkpoint_manager.restore(step,items=target)
        new_train_state=new_train_state['new_train_state']

    new_train_state = flax.jax_utils.replicate(new_train_state)

    for epoch in range(init_epoch, hp.train.epochs):

        loader = tqdm.tqdm(trainloader, desc='Loading train data')
        for data in loader:

            mel = data['mel'].numpy()
            pitch_label = data['pitch'].numpy()
            mel = jnp.asarray(mel)
            pitch_label = jnp.asarray(pitch_label)

            mel = shard(mel)
            pitch_label = shard(pitch_label)
            
            new_train_state,loss=combine_step(new_train_state,mel,pitch_label)

            step += 1

            loss = jax.device_get([loss[0]])
            if step % hp.log.info_interval == 0:
                logger.info("loss %.04f | step %d" % (loss[0], step))
                
        if epoch % hp.log.eval_interval == 0:
            metrics = validate(new_train_state,validation_dataset)
            val_loss = np.mean(metrics['loss'])
            rpa = np.mean(metrics['RPA'])
            rca = np.mean(metrics['RCA'])
            oa = np.mean(metrics['OA'])
            vr = np.mean(metrics['VR'])
            vfa = np.mean(metrics['VFA'])
            logger.info("val_loss %.04f RPA %.04f RCA %.04f OA %.04f VR %.04f VFA %.04f | step %d" % (val_loss,rpa,rca,oa,vr,vfa, step))
        if epoch % hp.log.save_interval == 0:
            new_train_state_s = flax.jax_utils.unreplicate(new_train_state)
            ckpt = {'new_train_state': new_train_state_s}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})
            del new_train_state_s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="configs/base.yaml",
                        help="yaml file for configuration")

    parser.add_argument('-n', '--name', type=str, default="sovits5.0",
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    assert hp.data.hop_length == 320, \
        'hp.data.hop_length must be equal to 320, got %d' % hp.data.hop_length

    train(args, args.checkpoint_path, hp)