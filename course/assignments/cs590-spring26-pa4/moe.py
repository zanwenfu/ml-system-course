import numpy as np

from mpiwrapper import mpi

def build_moe_init(input_dim, hidden_dim, output_dim, num_experts, seed=0):
    """
    Build a single canonical initialization for all MoE variants (Simple/EP/TP).
    Returns a dict: {"router": {"weight", "bias"}, "experts": [{"fc1": {"weight","bias"}, "fc2": {...}}, ...]}.
    All models loading from this get identical weights and produce identical outputs.
    """
    rng = np.random.RandomState(seed)
    init = {
        "router": {
            "weight": rng.randn(input_dim, num_experts).astype(np.float32) * 0.01,
            "bias": np.zeros(num_experts, dtype=np.float32),
        },
        "experts": [],
    }
    for _ in range(num_experts):
        init["experts"].append({
            "fc1": {
                "weight": rng.randn(input_dim, hidden_dim).astype(np.float32) * 0.01,
                "bias": np.zeros(hidden_dim, dtype=np.float32),
            },
            "fc2": {
                "weight": rng.randn(hidden_dim, output_dim).astype(np.float32) * 0.01,
                "bias": np.zeros(output_dim, dtype=np.float32),
            },
        })
    return init


class Linear:
    """Simple linear layer y = xW + b."""

    def __init__(self, in_features, out_features, weight, bias):
        self.weight = weight
        self.bias = np.asarray(bias, dtype=np.float32).copy()

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Expert network with one hidden layer and ReLU activation. No randomness: pass state or get zeros."""

    def __init__(self, input_dim, hidden_dim, output_dim, state):
        self.fc1 = Linear(input_dim, hidden_dim, weight=state["fc1"]["weight"], bias=state["fc1"]["bias"])
        self.fc2 = Linear(hidden_dim, output_dim, weight=state["fc2"]["weight"], bias=state["fc2"]["bias"])

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Routes inputs to experts using softmax-based gating. No randomness: pass weight/bias or get zeros."""

    def __init__(self, input_dim, num_experts, weight, bias):
        self.linear = Linear(input_dim, num_experts, weight, bias)

    def __call__(self, x, topk=1):
        logits = self.linear(x)

        # Softmax for routing probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Select top-k experts
        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)

        # Normalize gates to sum to 1
        gates = gates / np.sum(gates, axis=1, keepdims=True)

        return indices, gates


class ShardedLinear:
    """
    Linear layer that is sharded across processes
    Each process only holds a portion of the weight matrix
    
    Requires that out_features is evenly divisible by the world size
    """

    def __init__(self, in_features, out_features, weight, bias):
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        # Assert that out_features is evenly divisible by world_size
        assert out_features % self.world_size == 0, f"Output features ({out_features}) must be evenly divisible by world size ({self.world_size})"

        # Calculate the local output dimension
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size

        # Calculate output offset for this rank (simple with even division)
        self.output_offset = self.rank * self.local_out_features

        self.weight = weight[:, self.output_offset:self.output_offset + self.local_out_features]
        self.bias = bias[self.output_offset:self.output_offset + self.local_out_features]

    def __call__(self, x):
        # Compute local shard output
        local_output = np.dot(x, self.weight) + self.bias  # (batch, local_out)

        # Gather all shards
        gathered = mpi.allgather(local_output)

        # Concatenate along feature dimension
        full_output = np.concatenate(gathered, axis=1)

        return full_output.astype(np.float32)


class ShardedExpert:
    """Expert network with one hidden layer and ReLU activation, sharded across processes"""

    def __init__(self, input_dim, hidden_dim, output_dim, state=None):
        self.fc1 = ShardedLinear(input_dim, hidden_dim, weight=state["fc1"]["weight"], bias=state["fc1"]["bias"])
        self.fc2 = ShardedLinear(hidden_dim, output_dim, weight=state["fc2"]["weight"], bias=state["fc2"]["bias"])

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class MoE_TP:
    """
    Distributed Mixture of Experts using MPI for tensor parallelism
    
    TP-style MoE:
    - Each process holds a portion of every expert (sharded experts)
    - Router is replicated on all processes
    - All-to-all and all-gather communication patterns for processing
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Total number of experts in the model
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1, init_weights=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        self.router = Router(input_dim, num_experts, weight=init_weights["router"]["weight"], bias=init_weights["router"]["bias"])
        self.experts = [ShardedExpert(input_dim, hidden_dim, output_dim, state=init_weights["experts"][e]) for e in range(num_experts)]

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim), dtype=np.float32)

        # Routing (replicated)
        indices, gates = self.router(x, self.topk)

        # For each expert
        for expert_idx in range(self.num_experts):
            mask = (indices == expert_idx)

            if not np.any(mask):
                continue

            token_indices, topk_positions = np.where(mask)

            expert_inputs = x[token_indices]

            # Sharded expert forward (includes allgather inside)
            expert_outputs = self.experts[expert_idx](expert_inputs)

            # Vectorized accumulation
            expert_gates = gates[token_indices, topk_positions]
            outputs[token_indices] += expert_gates[:, None] * expert_outputs

        return outputs

    def __call__(self, x):
        return self.forward(x)


class SimpleMoE:
    """
    Simple reference implementation of Mixture of Experts.
    
    This class implements a basic MoE model that routes inputs to a subset
    of experts and combines their outputs using learned gating weights.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Number of expert networks
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1, init_weights=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        self.router = Router(input_dim, num_experts, weight=init_weights["router"]["weight"], bias=init_weights["router"]["bias"])
        self.experts = [Expert(input_dim, hidden_dim, output_dim, state=init_weights["experts"][e]) for e in range(num_experts)]

    def forward(self, x):
        """
        Forward pass through the MoE model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Get expert assignments and gates
        indices, gates = self.router(x, self.topk)
        
        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # Compute weighted combination of expert outputs
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i:i + 1]  # (1, input_dim)
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]

        return outputs

    def __call__(self, x):
        return self.forward(x)


class MoE_EP:
    """
    Distributed Mixture of Experts using MPI for expert parallelism
    
    EP-style MoE: 
    Each process hosts exactly one expert. Router is replicated on all processes.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1, init_weights=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts  # Total number of processes = number of experts
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.get_rank()

        self.router = Router(input_dim, self.num_experts, weight=init_weights["router"]["weight"], bias=init_weights["router"]["bias"])
        self.expert = Expert(input_dim, hidden_dim, output_dim, state=init_weights["experts"][self.rank])

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim), dtype=np.float32)

        indices, gates = self.router(x, self.topk)
        world_size = mpi.get_size()

        # --------------------------
        # 1. Send tokens to experts
        # --------------------------
        send_data = [[] for _ in range(world_size)]

        for i in range(batch_size):
            for k in range(self.topk):
                expert_rank = indices[i, k]
                gate = gates[i, k]
                send_data[expert_rank].append((i, gate, x[i]))

        received = mpi.alltoall(send_data)

        # Flatten received tokens for this expert
        local_items = [item for sublist in received for item in sublist]

        # --------------------------
        # 2. Run local expert
        # --------------------------
        return_data = [[] for _ in range(world_size)]

        if local_items:
            token_indices = [item[0] for item in local_items]
            gates_local = [item[1] for item in local_items]
            inputs_local = np.stack([item[2] for item in local_items], axis=0)

            expert_outputs = self.expert(inputs_local)

            # Send results back only to originating ranks
            for idx, token_idx in enumerate(token_indices):
                source_rank = idx // 1  # placeholder not used logically
                return_data = return_data  # kept structure intact

            # Instead, we return results to the rank that sent them
            # We know received is structured per source rank
            offset = 0
            for src_rank, sublist in enumerate(received):
                count = len(sublist)
                for j in range(count):
                    token_idx, gate, _ = sublist[j]
                    output_vec = expert_outputs[offset + j]
                    return_data[src_rank].append((token_idx, gate, output_vec))
                offset += count

        gathered = mpi.alltoall(return_data)

        # --------------------------
        # 3. Accumulate
        # --------------------------
        for sublist in gathered:
            for token_idx, gate, output_vec in sublist:
                outputs[token_idx] += gate * output_vec

        return outputs

    def __call__(self, x):
        return self.forward(x)
