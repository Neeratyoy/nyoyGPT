"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
from pathlib import Path
import pickle
from contextlib import nullcontext
from typing import Any, Callable, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from utils import (
    ddp_setup, get_custom_dataloader, get_full_hp_list, prepare_model_optimizer
)


####################
# HELPER FUNCTIONS #
####################

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model: Any, hps: dict, get_batch: Callable, ctx: Any):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hps["eval_iters"])
        for k in range(hps["eval_iters"]):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, hps):
    # 1) linear warmup for warmup_iters steps
    if it < hps["warmup_iters"]:
        return hps["learning_rate"] * it / hps["warmup_iters"]
    # 2) if it > lr_decay_iters, return min learning rate
    if it > hps["lr_decay_iters"]:
        return hps["min_lr"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - hps["warmup_iters"]) / (hps["lr_decay_iters"] - hps["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return hps["min_lr"] + coeff * (hps["learning_rate"] - hps["min_lr"])


def main_run(**kwargs):
    get_hps = get_full_hp_list()
    # overwriting for a smaller scale test setup
    hps = get_hps(**kwargs)

    breakpoint()

    # setup PyTorch DDP
    (
        ddp,
        device,
        dtype,
        compile,
        config,
        master_process,
        seed_offset,
        ddp_world_size,
        ddp_local_rank,
        tokens_per_iter
    ) = ddp_setup(hps)

    # uncomment for Mac
    device = "cpu"
    compile = False

    if master_process:
        os.makedirs(hps["out_dir"], exist_ok=True)
    torch.manual_seed(hps["seed"] + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = (
        nullcontext() 
        if device_type == 'cpu' 
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    get_batch, meta_vocab_size = get_custom_dataloader(
        hps["dataset"],
        hps["block_size"],
        hps["batch_size"],
        device,
        device_type,
        hps["data_path"],
    )

    out_dir = hps["out_dir"]  # Path(__file__).resolve().parent / "out"
    model_args, model, optimizer, scaler = prepare_model_optimizer(
        hps=hps,
        ddp=ddp,
        device=device,
        device_type=device_type,
        ddp_local_rank=ddp_local_rank,
        compile=compile,
        dtype=dtype,
        init_from=hps["init_from"], 
        meta_vocab_size=meta_vocab_size,
        out_dir=out_dir,
    )

    # training loop
    X, Y = get_batch('train')  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, hps) if hps["decay_lr"] else hps["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % hps["eval_interval"] == 0 and master_process:
            losses = estimate_loss(model, hps, get_batch, ctx)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            if losses['val'] < best_val_loss or hps["always_save_checkpoint"]:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and hps["eval_only"]:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(hps["gradient_accumulation_steps"]):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == hps["gradient_accumulation_steps"] - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / hps["gradient_accumulation_steps"]  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if hps["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hps["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % hps["log_interval"] == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * hps["gradient_accumulation_steps"]
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    hps["batch_size"] * hps["gradient_accumulation_steps"], dt
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(
                f"iter {iter_num}: loss {lossf:.4f}, "
                f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > hps["max_iters"]:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main_run(
        block_size=256,
        n_layer=6,
        n_head=8,
        n_embd=256,
        max_iters=5000,
        warmup_iters=int(5000 * 0.05),  # 5 % of total training steps
        lr_decay_iters=5000,
        eval_interval=100,
        log_interval=10,
        eval_iters=200,
    )
