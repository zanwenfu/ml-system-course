from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(self, node_A: Node, node_B: Node, normalized_shape: List[int], eps: float = 1e-5) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 2
        A, B = input_values
        X = A @ B
        eps = node.attrs["eps"]

        k = len(node.attrs["normalized_shape"])
        if k == 0:
            return X

        dims = tuple(range(X.dim() - k, X.dim()))
        mu = X.mean(dim=dims, keepdim=True)
        var = ((X - mu) ** 2).mean(dim=dims, keepdim=True)
        return (X - mu) / torch.sqrt(var + eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        A, B = node.inputs
        eps = node.attrs["eps"]
        normalized_shape = node.attrs["normalized_shape"]

        X = matmul(A, B)

        k = len(normalized_shape)
        dims = tuple(range(-k, 0))

        mu = mean(X, dim=dims, keepdim=True)
        xmu = X - mu
        var = mean(power(xmu, 2.0), dim=dims, keepdim=True)
        invstd = ones_like(X) / sqrt(var + eps)
        y = xmu * invstd

        dy = output_grad
        sum_dy = sum_op(dy, dim=dims, keepdim=True)
        sum_dy_y = sum_op(dy * y, dim=dims, keepdim=True)
        N = sum_op(ones_like(X), dim=dims, keepdim=True)

        dX = ((dy * N) - sum_dy - (y * sum_dy_y)) * invstd / N

        Bt = transpose(B, -1, -2)
        At = transpose(A, -1, -2)
        dA = matmul(dX, Bt)
        dB = matmul(At, dX)
        return [dA, dB]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self,
        node_A: Node,
        node_B: Node,
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        A, B = input_values
        X = A @ B
        return torch.softmax(X, dim=node.attrs["dim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A, B = node.inputs
        dim = node.attrs["dim"]

        # X = A @ B
        X = matmul(A, B)

        # Softmax backward: dX = S * (dY - sum(dY*S, dim, keepdim=True))
        S = softmax(X, dim=dim)
        dot = sum_op(output_grad * S, dim=(dim,), keepdim=True)
        dX = S * (output_grad - dot)

        # MatMul backward
        Bt = transpose(B, -1, -2)
        At = transpose(A, -1, -2)
        dA = matmul(dX, Bt)
        dB = matmul(At, dX)
        return [dA, dB]


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
