from typing import Any, Dict, List

import torch


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]
    
class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]

class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]
    
class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.ones_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class SumOp(Op):
    """
    Op to compute sum along specified dimensions.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].sum(dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]

        if keepdim:
            return [output_grad]

        # Key fix:
        # - For loss: sum over dim=(0,) on a (B,) vector -> scalar gradient expands back fine with expand_as
        # - For other cases (e.g. (B,C)->(B,) or (B,T,D)->(B,D)), we typically need to INSERT a dim of size 1.
        #   expand_as can't insert dims, so use expand_as_3d.
        if dim == (0,):
            return [expand_as(output_grad, node.inputs[0])]
        else:
            return [expand_as_3d(output_grad, node.inputs[0])]


class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    This op assumes PyTorch-style expand rules: it only works when
    input already has singleton dims in the right positions.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # Without static shape metadata, we implement the conservative version:
        # sum along dim=0 (this matches the common "batch broadcast" usage).
        return [sum_op(output_grad, dim=(0,), keepdim=False), zeros_like(output_grad)]


class ExpandAsOp3d(Op):
    """
    A specialized "expand" helper that INSERTS a singleton dimension at dim=1
    (the middle dimension), then expands.

    This is exactly what we need for:
      - (B,)   -> (B, C)      by unsqueeze(1)
      - (B, D) -> (B, T, D)   by unsqueeze(1)
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast3d({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values

        # If already same rank, just expand (will work if broadcastable)
        if input_tensor.dim() == target_tensor.dim():
            return input_tensor.expand_as(target_tensor)

        # We only support inserting dims at position 1 (the needed cases above).
        # This avoids the "Dimension out of range" crash from unsqueeze(1) on scalars.
        if input_tensor.dim() == 0:
            # For scalar -> anything, we can safely reshape to all-ones then expand.
            base = input_tensor.view(*([1] * target_tensor.dim()))
            return base.expand_as(target_tensor)

        # Insert missing dims at dim=1 until ranks match
        out = input_tensor
        while out.dim() < target_tensor.dim():
            out = out.unsqueeze(1)
        return out.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # Because we inserted singleton dims at dim=1, we must sum over dim=1
        # to collapse back to the original shape.
        return [sum_op(output_grad, dim=(1,), keepdim=False), zeros_like(output_grad)]


class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(self, node_A: Node, input_shape: List[int], target_shape: List[int]) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.attrs["target_shape"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.
        
        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError("Input shape is not set. Make sure compute() is called before gradient()")
            
        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]
        
        dims_to_sum = []
        for i, (in_size, out_size) in enumerate(zip(input_shape[::-1], output_shape[::-1])):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)
                
        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)
            
        if len(output_shape) > len(input_shape):
            grad = sum_op(grad, dim=list(range(len(output_shape) - len(input_shape))), keepdim=False)
            
        return [grad]

class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        return input_values[0] / input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """d(A/B) = dA*(1/B) + dB*(-A/B^2)"""
        A, B = node.inputs
        grad_A = output_grad / B
        grad_B = mul_by_const(output_grad, -1) * (A / (B * B))
        return [grad_A, grad_B]
    
class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] / node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """d(x/c) = (1/c) dx"""
        return [output_grad / node.constant]


class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].transpose(node.dim0, node.dim1)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Transpose is its own inverse: d/dx transpose(x, a,b) = transpose(dy, a,b)."""
        return [transpose(output_grad, node.dim0, node.dim1)]

class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 2
        return input_values[0] @ input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        A, B = node.inputs
        # dA = dY @ B^T
        # dB = A^T @ dY
        Bt = transpose(B, -1, -2)
        At = transpose(A, -1, -2)
        grad_A = matmul(output_grad, Bt)
        grad_B = matmul(At, output_grad)
        return [grad_A, grad_B]


class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.softmax(input_values[0], dim=node.dim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        For s = softmax(x):
        dx = s * (dy - sum(dy*s, dim, keepdim=True))
        """
        x = node.inputs[0]
        dim = node.dim
        s = softmax(x, dim=dim)
        dot = sum_op(output_grad * s, dim=(dim,), keepdim=True)
        return [s * (output_grad - dot)]

class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(self, node_A: Node, normalized_shape: List[int], eps: float = 1e-5) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """
        IMPORTANT:
        Our framework treats `normalized_shape` as defining ONLY how many trailing dims
        to normalize over (k = len(normalized_shape)), not enforcing exact sizes.
        This matches our LayerNormOp.gradient implementation.
        """
        assert len(input_values) == 1
        x = input_values[0]
        eps = node.eps

        k = len(node.normalized_shape)
        if k == 0:
            return x

        dims = tuple(range(x.dim() - k, x.dim()))
        mu = x.mean(dim=dims, keepdim=True)
        var = ((x - mu) ** 2).mean(dim=dims, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Standard LayerNorm backward (no gamma/beta):
        Let dims be the last k dims where k=len(normalized_shape).
        y = (x - mu) / sqrt(var + eps)
        dx = (1/N) * invstd * (N*dy - sum(dy) - y*sum(dy*y))
        where sums are over dims, keepdim=True, and N is count of reduced elems.
        """
        x = node.inputs[0]
        eps = node.eps
        k = len(node.normalized_shape)
        dims = tuple(range(-k, 0))  # normalize over last k dims (works with negative dims)

        mu = mean(x, dim=dims, keepdim=True)
        xmu = x - mu
        var = mean(power(xmu, 2.0), dim=dims, keepdim=True)
        invstd = ones_like(x) / sqrt(var + eps)
        y = xmu * invstd

        dy = output_grad
        sum_dy = sum_op(dy, dim=dims, keepdim=True)
        sum_dy_y = sum_op(dy * y, dim=dims, keepdim=True)
        N = sum_op(ones_like(x), dim=dims, keepdim=True)

        dx = ((dy * N) - sum_dy - (y * sum_dy_y)) * invstd / N
        return [dx]
    

class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.relu(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = node.inputs[0]
        mask = greater(x, zeros_like(x))  # 1 where x>0 else 0
        return [output_grad * mask]



class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Sqrt({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.sqrt(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = node.inputs[0]
        denom = mul_by_const(sqrt(x), 2.0)   # 2*sqrt(x)
        return [output_grad / denom]


class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.pow(input_values[0], node.exponent)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = node.inputs[0]
        p = node.exponent
        # d(x^p) = p * x^(p-1)
        return [output_grad * mul_by_const(power(x, p - 1.0), p)]


class MeanOp(Op):
    """Op to compute mean along specified dimensions."""

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].mean(dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Gradient distributes evenly:
        d/dx mean(x over dim) = broadcast(output_grad / count)
        count can be built dynamically as sum(ones_like(x)) over same dims.
        """
        x = node.inputs[0]
        dim = node.dim
        keepdim = node.keepdim

        # count has the same shape as mean output (with keepdim behavior)
        count = sum_op(ones_like(x), dim=dim, keepdim=keepdim)

        if keepdim:
            # output_grad shape matches count; broadcast to x
            g = output_grad / count
            return [expand_as(g, x)]
        else:
            # use the same "expand" convention as SumOp.gradient in this file
            g = expand_as_3d(output_grad, x)
            c = expand_as_3d(count, x)
            return [g / c]

# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
broadcast = BroadcastOp()

def topological_sort(nodes):
    """Return nodes in topological order (inputs before outputs)."""
    if isinstance(nodes, Node):
        nodes = [nodes]

    visited = set()
    order: List[Node] = []

    def dfs(n: Node):
        if n in visited:
            return
        visited.add(n)
        for inp in n.inputs:
            dfs(inp)
        order.append(n)

    for n in nodes:
        dfs(n)

    return order


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:
        # 1) topo order of everything needed
        topo = topological_sort(self.eval_nodes)

        # 2) forward compute table
        node_to_val: Dict[Node, torch.Tensor] = {}

        for n in topo:
            # placeholder / Variable nodes must be provided
            if n.op is placeholder:
                if n not in input_values:
                    raise ValueError(f"Missing value for input node: {n}")
                node_to_val[n] = input_values[n]
                continue

            # compute from inputs
            in_vals = [node_to_val[inp] for inp in n.inputs]
            node_to_val[n] = n.op.compute(n, in_vals)

        # 3) return requested values in order
        return [node_to_val[n] for n in self.eval_nodes]



def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Build backward graph for d(output_node)/d(nodes[i])."""
    topo = topological_sort(output_node)
    rev_topo = topo[::-1]

    # node -> list of gradient contributions (Nodes)
    node_to_grads: Dict[Node, List[Node]] = {output_node: [ones_like(output_node)]}

    def sum_grads(grads: List[Node]) -> Node:
        """Sum a list of Node gradients into one Node."""
        assert len(grads) > 0
        out = grads[0]
        for g in grads[1:]:
            out = out + g
        return out

    for n in rev_topo:
        if n not in node_to_grads:
            continue

        out_grad = sum_grads(node_to_grads[n])

        # placeholders have no parents to propagate to
        if n.op is placeholder:
            continue

        in_grads = n.op.gradient(n, out_grad)
        assert len(in_grads) == len(n.inputs)

        for inp, g in zip(n.inputs, in_grads):
            if inp not in node_to_grads:
                node_to_grads[inp] = []
            node_to_grads[inp].append(g)

    # Return grads for requested nodes; if disconnected, return zeros_like
    result = []
    for n in nodes:
        if n in node_to_grads:
            result.append(sum_grads(node_to_grads[n]))
        else:
            result.append(zeros_like(n))
    return result

