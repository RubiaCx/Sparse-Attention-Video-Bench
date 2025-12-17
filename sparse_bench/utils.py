import os

import torch
import torch.distributed as dist
import time


def is_distributed():
    return os.environ.get("RANK") is not None


def is_rank_zero():
    return dist.get_rank() == 0


def init_distributed():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()