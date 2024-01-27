import numpy as np
import os
from pathlib import Path
import pickle 
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Callable, Tuple

from model import GPTConfig, GPT


def ddp_setup(hps: dict):
    # DDP settings
    backend = 'nccl'  # 'nccl', 'gloo', etc.
    # system
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = (
        'bfloat16' 
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() 
        else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    )
    compile = False  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [
        k for k,v in globals().items() 
        if not k.startswith('_') and isinstance(v, (int, float, bool, str))
    ]
    exec(open('configurator.py').read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert hps["gradient_accumulation_steps"] % ddp_world_size == 0
        hps["gradient_accumulation_steps"] //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = None
    tokens_per_iter = hps["gradient_accumulation_steps"] * ddp_world_size * hps["batch_size"] * hps["block_size"]
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    return (
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
    )


def prepare_model_optimizer(
    hps: dict,
    ddp: bool,
    device: str,
    device_type: str,
    ddp_local_rank: int,
    compile: bool = False,
    dtype: str = "bfloat16",
    init_from: str = "scratch",
    meta_vocab_size: int | None = None,
    load_dir: str | Path | None = None,
    out_dir: str | Path = "./out",
):
    iter_num = 0
    best_val_loss = 1e-9
    # model init
    model_args = dict(
        n_layer=hps["n_layer"],
        n_head=hps["n_head"],
        n_embd=hps["n_embd"],
        block_size=hps["block_size"],
        bias=hps["bias"],
        vocab_size=None,
        dropout=hps["dropout"],
    ) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        # resume training from a checkpoint.
        ckpt_path = (
            os.path.join(out_dir, 'ckpt.pt')
            if load_dir is None
            else os.path.join(load_dir, 'ckpt.pt')
        )
        print(f"Resuming training from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=hps["dropout"])
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if hps["block_size"] < model.config.block_size:
        model.crop_block_size(hps["block_size"])
        model_args['block_size'] = hps["block_size"] # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(
        hps["weight_decay"],
        hps["learning_rate"],
        (hps["beta1"], hps["beta2"]),
        device_type
    )
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    return model_args, model, optimizer, scaler, iter_num, best_val_loss


##########################################
# poor man's data loader (from Karpathy) #
##########################################

def get_custom_dataloader(
    dataset: str,
    block_size: int,
    batch_size: int,
    device: str,
    device_type: str,
    data_dir: str | Path,
) -> Tuple[Callable, str]:
    data_dir = os.path.join(data_dir, dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([
            torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix
        ])
        y = torch.stack([
            torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix
        ])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = (
                x.pin_memory().to(device, non_blocking=True), 
                y.pin_memory().to(device, non_blocking=True)
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
    return get_batch, meta_vocab_size


#############################
# POTENTIAL HYPERPARAMETERS #
#############################

def get_full_hp_list():
    hps = dict(
        seed = 1337,
        # dataset related entries
        dataset = "shakespeare_char",
        data_path = "./data",  # Path(__file__).resolve().parent / "data",
        # model training knobs
        gradient_accumulation_steps = 5 * 8,  # used to simulate larger batch sizes
        batch_size = 12,  # if gradient_accumulation_steps > 1, this is the micro-batch size
        block_size = 1024,
        # model
        n_layer = 12,
        n_head = 12,
        n_embd = 768,
        dropout = 0.0, # for pretraining 0 is good, for finetuning try 0.1+
        bias = False, # do we use bias inside LayerNorm and Linear layers?
        # adamw optimizer
        learning_rate = 6e-4, # max learning rate
        max_iters = 600000, # total number of training iterations
        step_budget = 600000,
        weight_decay = 1e-1,
        beta1 = 0.9,
        beta2 = 0.95,
        grad_clip = 1.0, # clip gradients at this value, or disable if == 0.0
        # learning rate decay settings
        decay_lr = True, # whether to decay the learning rate
        warmup_iters = 2000, # how many steps to warm up for
        lr_decay_iters = 600000, # should be ~= max_iters per Chinchilla
        min_lr = 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        # training stuff
        load_dir = None,
        out_dir = "out",
        eval_interval = 100,
        log_interval = 10,
        eval_iters = 200,
        eval_only = False,  # if True, script exits right after the first eval
        always_save_checkpoint = True,  # if True, always save a checkpoint after each eval
        init_from = "scratch",  # 'scratch' or 'resume' or 'gpt2*'
    )
    def get_custom_hp_list(**kwargs):
        nonlocal hps
        _hp_diff = set(kwargs.keys()) - set(hps.keys())
        assert len(_hp_diff) == 0, f"Hyperparameters not recognized: {_hp_diff}"
        hps.update(kwargs)
        return hps
    return get_custom_hp_list
