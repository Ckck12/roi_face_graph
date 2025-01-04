# src/engine/ddp_utils.py

import torch.distributed as dist
import torch.multiprocessing as mp

def init_distributed_mode(rank, world_size, backend, dist_url):
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

def cleanup_distributed():
    dist.destroy_process_group()

def run_ddp(fn, world_size, backend, dist_url, args):
    mp.spawn(
        fn,
        nprocs=world_size,
        args=(world_size, backend, dist_url, args),
        join=True
    )
