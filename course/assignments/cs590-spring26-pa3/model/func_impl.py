import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.
    """

    # ----- Compute indices -----
    dp_idx = rank // mp_size
    mp_idx = rank % mp_size

    # ----- Create communicators -----
    # Model Parallel communicator (within same DP replica)
    mp_comm = comm.Split(color=dp_idx, key=mp_idx)

    # Data Parallel communicator (same shard across replicas)
    dp_comm = comm.Split(color=mp_idx, key=dp_idx)

    # ----- Partition dimensions -----
    if fc_layer in ["fc_q", "fc_k", "fc_v"]:
        # Partition along output dimension
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size

    elif fc_layer == "fc_o":
        # Partition along input dimension
        part_in_dim = in_dim // mp_size
        part_out_dim = out_dim

    else:
        raise ValueError(f"Unsupported fc_layer type: {fc_layer}")

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim


def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Gather fc_o forward inputs across MP nodes.

    Each node holds:
        (batch_size, seq_length, part_in_dim)

    Return:
        (batch_size, seq_length, part_in_dim * mp_size)
    """
    x = np.ascontiguousarray(x)

    gathered_list = mp_comm.allgather(x)
    collected_x = np.concatenate(gathered_list, axis=-1)

    return collected_x


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Gather fc_o forward outputs across MP nodes.

    Each node holds:
        (batch_size, seq_length, part_out_dim)

    Return:
        (batch_size, seq_length, part_out_dim * mp_size)
    """
    out = np.ascontiguousarray(out)

    gathered_list = mp_comm.allgather(out)
    collected_out = np.concatenate(gathered_list, axis=-1)

    return collected_out


def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Split full output_grad along output dimension for local MP node.
    """
    batch_size, seq_length, out_dim = output_grad.shape
    part_out_dim = out_dim // mp_size

    start = mp_group_idx * part_out_dim
    end = start + part_out_dim

    collected_output_grad = output_grad[:, :, start:end]

    return collected_output_grad


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Reduce and scatter grad_x across MP nodes.

    Steps:
        1. Sum grad_x across MP nodes (Allreduce).
        2. Slice along input dimension.
    """
    grad_x = np.ascontiguousarray(grad_x)

    # Step 1: Sum across nodes
    reduced_grad_x = np.zeros_like(grad_x)
    mp_comm.Allreduce(grad_x, reduced_grad_x, op=MPI.SUM)

    # Step 2: Slice along input dimension
    batch_size, seq_length, in_dim = reduced_grad_x.shape
    part_in_dim = in_dim // mp_size

    rank = mp_comm.Get_rank()
    start = rank * part_in_dim
    end = start + part_in_dim

    collected_grad_x = reduced_grad_x[:, :, start:end]

    return collected_grad_x
