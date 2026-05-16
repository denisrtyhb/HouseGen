"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 4

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    ## temporary removed to manually set the CUDA_VISIBLE_DEVICES
    #os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    rank = MPI.COMM_WORLD.Get_rank()

    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    print(f"[load_state_dict] rank={rank} after chunk_size={chunk_size}", flush=True)

    if rank == 0:
        print(f"[load_state_dict] rank={rank} entering rank-0 branch", flush=True)

        print(f"[load_state_dict] rank={rank} before BlobFile open path={path!r}", flush=True)
        with bf.BlobFile(path, "rb") as f:
            print(f"[load_state_dict] rank={rank} after BlobFile opened", flush=True)
            data = f.read()
            print(f"[load_state_dict] rank={rank} after f.read() len(data)={len(data)}", flush=True)
        print(f"[load_state_dict] rank={rank} after with block (file closed)", flush=True)

        num_chunks = len(data) // chunk_size
        print(f"[load_state_dict] rank={rank} after num_chunks = len(data)//chunk_size -> {num_chunks}", flush=True)
        if len(data) % chunk_size:
            num_chunks += 1
            print(f"[load_state_dict] rank={rank} after bump num_chunks -> {num_chunks}", flush=True)

        MPI.COMM_WORLD.bcast(num_chunks)
        print(f"[load_state_dict] rank={rank} after bcast(num_chunks)", flush=True)

        for i in range(0, len(data), chunk_size):
            print(f"[load_state_dict] rank={rank} before bcast chunk i={i} size={len(data[i:i+chunk_size])}", flush=True)
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
            print(f"[load_state_dict] rank={rank} after bcast chunk i={i}", flush=True)

        print(f"[load_state_dict] rank={rank} finished broadcasting all chunks", flush=True)
    else:
        print(f"[load_state_dict] rank={rank} entering non-root branch", flush=True)

        num_chunks = MPI.COMM_WORLD.bcast(None)
        print(f"[load_state_dict] rank={rank} after recv num_chunks={num_chunks}", flush=True)

        data = bytes()
        print(f"[load_state_dict] rank={rank} after data = bytes()", flush=True)

        for chunk_idx in range(num_chunks):
            print(f"[load_state_dict] rank={rank} before bcast(chunk {chunk_idx+1}/{num_chunks}) len(data)={len(data)}", flush=True)
            data += MPI.COMM_WORLD.bcast(None)
            print(f"[load_state_dict] rank={rank} after bcast(chunk {chunk_idx+1}/{num_chunks}) len(data)={len(data)}", flush=True)

        print(f"[load_state_dict] rank={rank} finished receiving all chunks", flush=True)

    print(f"[load_state_dict] rank={rank} before th.load len(data)={len(data)}", flush=True)
    _blob_n = len(data)
    try:
        import psutil

        _rss_b = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        print(
            f"[mem] rank={rank} checkpoint serialized blob {_blob_n} bytes "
            f"({_blob_n / (1024**3):.4f} GiB); RSS before torch.load ~{_rss_b:.4f} GiB",
            flush=True,
        )
    except Exception:
        print(
            f"[mem] rank={rank} checkpoint serialized blob {_blob_n} bytes "
            f"({_blob_n / (1024**3):.4f} GiB) — pip install psutil for RSS",
            flush=True,
        )
    out = th.load(io.BytesIO(data), **kwargs)
    del data
    try:
        import psutil

        _rss_a = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        print(
            f"[mem] rank={rank} RSS after torch.load (~unpacked state on CPU ranks): ~{_rss_a:.4f} GiB",
            flush=True,
        )
    except Exception:
        pass
    print(f"[load_state_dict] rank={rank} after th.load", flush=True)
    return out


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
