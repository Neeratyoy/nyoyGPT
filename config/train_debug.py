# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 256
gradient_accumulation_steps = 5 * 8

max_iters = 5000
lr_decay_iters = max_iters  # could multiply 1.05 based on some scaling papers

# model
n_layer = 6
n_head = 8
n_embd = 256

learning_rate = 6e-4
decay_lr = True
warmup_iters = int(max_iters * 0.05)  # 5% of total steps for LR warmup
min_lr = 6e-5

# eval stuff
eval_interval = 100
eval_iters = 100
log_interval = 10

# weight decay
weight_decay = 1e-1
