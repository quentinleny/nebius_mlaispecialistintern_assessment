# nccl_allreduce.py

import os
import socket
import torch
import torch.distributed as dist

def main():
    print(f"host={socket.gethostname()}")
    print(f"RANK={os.environ.get('RANK')}")
    print(f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    x = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    expected = world_size * (world_size + 1) / 2

    print(
        f"rank={rank} local_rank={local_rank} "
        f"device={torch.cuda.get_device_name(local_rank)} "
        f"all_reduce={x.item()} expected={expected}"
    )

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
