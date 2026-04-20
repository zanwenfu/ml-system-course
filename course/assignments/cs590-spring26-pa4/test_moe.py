import numpy as np
from mpi4py import MPI
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP, build_moe_init
import time

def run_moe(
    moe_type="tp",
    batch_size=8,
    feature_dim=32,
    hidden_dim=128,
    output_dim=64,
    num_experts=None,
    topk=2,
    init_weights=None,
):
    """
    Run MoE forward. If init_weights is provided, all variants load from it (identical outputs).
    """
    num_experts = num_experts if num_experts is not None else mpi.get_size()

    input_seed = 42
    if moe_type == "simple":
        X = np.random.RandomState(input_seed).randn(batch_size, feature_dim).astype(np.float32)
    else:
        if mpi.get_rank() == 0:
            X = np.random.RandomState(input_seed).randn(batch_size, feature_dim).astype(np.float32)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)

    model_class = {"simple": SimpleMoE, "ep": MoE_EP, "tp": MoE_TP}.get(moe_type, MoE_TP)
    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk,
        init_weights=init_weights,
    )
    
    # Run forward pass
    # Warm up
    _ = moe(X)
    
    # Measure time
    N = 3
    start_time = time.time()
    for _ in range(N):
        outputs = moe(X)
    end_time = time.time()
    avg_duration_ms = 1000 * (end_time - start_time) / N
    
    # Print timing information
    if mpi.get_rank() == 0:
        print(f"Forward pass time for {moe_type} MoE: {avg_duration_ms} ms")

    return dict(
        outputs=outputs,
        avg_duration_ms=avg_duration_ms
    )
    
    
BATCH_SIZE = 10
FEATURE_DIM = 10
HIDDEN_DIM = 10
OUTPUT_DIM = 10
TOPK = 10
INIT_SEED = 0

def test_simple_moe():
    num_experts = mpi.get_size()
    init = build_moe_init(FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM, num_experts, seed=INIT_SEED)
    result = run_moe(
        "simple",
        batch_size=BATCH_SIZE,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_experts=num_experts,
        topk=TOPK,
        init_weights=init,
    )
    output = result["outputs"]
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    return output


def test_ep_moe():
    num_experts = mpi.get_size()
    init = build_moe_init(FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM, num_experts, seed=INIT_SEED)
    result = run_moe(
        "ep",
        batch_size=BATCH_SIZE,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_experts=num_experts,
        topk=TOPK,
        init_weights=init,
    )
    output = result["outputs"]
    return output


def test_tp_moe():
    num_experts = mpi.get_size()
    init = build_moe_init(FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM, num_experts, seed=INIT_SEED)
    result = run_moe(
        "tp",
        batch_size=BATCH_SIZE,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_experts=num_experts,
        topk=TOPK,
        init_weights=init,
    )
    output = result["outputs"]
    return output


if __name__ == "__main__":
    num_experts = mpi.get_size()
    simple_output = test_simple_moe()
    ep_output = test_ep_moe()
    tp_output = test_tp_moe()
    if mpi.get_rank() == 0:
        assert np.allclose(simple_output, ep_output), "Simple vs EP mismatch"
        assert np.allclose(simple_output, tp_output), "Simple vs TP mismatch"
        print("All MoE tests passed ✅")