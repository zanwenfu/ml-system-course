import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def transformer(
    X: ad.Node,
    nodes: List[ad.Node],
    model_dim: int,
    seq_length: int,
    eps,
    batch_size,
    num_classes,
) -> ad.Node:
    """
    Build a single-layer Transformer forward graph (single-head, no residuals).
    Output: (batch_size, num_classes)
    """

    # Unpack parameters
    # Shapes expected:
    # W_Q/W_K/W_V: (input_dim, model_dim)
    # W_O:         (model_dim, model_dim)
    # W_1:         (model_dim, model_dim)
    # W_2:         (model_dim, num_classes)
    # b_1:         (1, 1, model_dim)   (recommended)
    # b_2:         (1, num_classes)    (recommended)
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # -------------------------
    # Part 2: Single-head Attention
    # (Part 1 is used here as linear projections)
    # -------------------------
    Q = ad.matmul(X, W_Q)                      # (B, T, D)
    K = ad.matmul(X, W_K)                      # (B, T, D)
    V = ad.matmul(X, W_V)                      # (B, T, D)

    Kt = ad.transpose(K, -1, -2)               # (B, D, T)
    scores = ad.matmul(Q, Kt)                  # (B, T, T)

    # scale by sqrt(dk)
    dk_sqrt = float(model_dim) ** 0.5          # python float
    scores = scores / dk_sqrt                  # DivByConstOp

    A = ad.softmax(scores, dim=-1)             # (B, T, T)
    attn = ad.matmul(A, V)                     # (B, T, D)

    # output projection
    attn = ad.matmul(attn, W_O)                # (B, T, D)

    # LayerNorm after attention (no residual per assignment)
    attn = ad.layernorm(attn, normalized_shape=[model_dim], eps=eps)  # (B, T, D)

    # -------------------------
    # Part 3: Encoder feed-forward
    # -------------------------
    ff = ad.matmul(attn, W_1)                  # (B, T, D)

    # add bias b_1 (broadcast to B,T,D)
    b1_bt = ad.broadcast(b_1, input_shape=[1, 1, model_dim],
                         target_shape=[batch_size, seq_length, model_dim])
    ff = ff + b1_bt

    ff = ad.relu(ff)                           # (B, T, D)

    # (Optional LN after FFN; helps training a lot even without residuals)
    ff = ad.layernorm(ff, normalized_shape=[model_dim], eps=eps)       # (B, T, D)

    # -------------------------
    # Sequence pooling + classifier head
    # -------------------------
    # mean over sequence dimension -> (B, D)
    pooled = ad.mean(ff, dim=(1,), keepdim=False)  # mean over T

    logits = ad.matmul(pooled, W_2)                # (B, C)

    # add bias b_2 (broadcast to B,C)
    b2_b = ad.broadcast(b_2, input_shape=[1, num_classes],
                        target_shape=[batch_size, num_classes])
    logits = logits + b2_b

    return logits



def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this task, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    """TODO: Your code here"""
    # Probabilities: (B, C)
    P = ad.softmax(Z, dim=-1)

    # Probability assigned to the true class: (B,)
    # sum over classes dimension
    p_true = ad.sum_op(P * y_one_hot, dim=(1,), keepdim=False)

    # Negative log likelihood: (B,)
    nll = ad.log(p_true) * (-1.0)

    # Average over batch -> scalar ()
    loss = ad.sum_op(nll, dim=(0,), keepdim=False) / float(batch_size)
    return loss



def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_loss = 0.0

    for i in range(num_batches):
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue
        end_idx = min(start_idx + batch_size, num_examples)

        X_batch = X[start_idx:end_idx, :max_len]   # (B, 28, 28)
        y_batch = y[start_idx:end_idx]             # (B, 10) one-hot

        # Run forward+backward through the evaluator
        logits, loss_val, *grads = f_run_model(X_batch, y_batch, model_weights)

        # Update parameters (SGD)
        # grads list aligned with model_weights list
        for j in range(len(model_weights)):
            model_weights[j] = model_weights[j] - lr * grads[j]

        # Accumulate loss (loss_val is a 0-dim tensor or scalar tensor)
        total_loss += float(loss_val) * (end_idx - start_idx)

    average_loss = total_loss / num_examples
    print("Avg_loss:", average_loss)
    return model_weights, average_loss

def train_model():
    # -------------------------
    # Hyperparameters
    # -------------------------
    input_dim = 28          # each token is a row of MNIST
    seq_length = max_len    # 28
    num_classes = 10
    model_dim = 128
    eps = 1e-5

    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # -------------------------
    # Build forward/backward graph
    # -------------------------
    X_node = ad.Variable(name="X")   # (B, T, input_dim)
    y_node = ad.Variable(name="y")   # (B, C) one-hot

    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")    # will feed as (1,1,D)
    b_2 = ad.Variable(name="b_2")    # will feed as (1,C)

    param_nodes: List[ad.Node] = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_predict: ad.Node = transformer(
        X_node,
        param_nodes,
        model_dim=model_dim,
        seq_length=seq_length,
        eps=eps,
        batch_size=batch_size,
        num_classes=num_classes,
    )

    loss: ad.Node = softmax_loss(y_predict, y_node, batch_size)

    grads: List[ad.Node] = ad.gradients(loss, param_nodes)

    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # -------------------------
    # Load MNIST
    # -------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    X_train_np = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_train_np = train_dataset.targets.numpy()

    X_test_np = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_test_np = test_dataset.targets.numpy()

    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh_np = encoder.fit_transform(y_train_np.reshape(-1, 1))

    # torch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_oh_np, dtype=torch.float32)
    # keep y_test as numpy ints for accuracy compare
    # y_test_np stays as is

    # -------------------------
    # Initialize weights
    # -------------------------
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)

    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val, dtype=torch.float32),
        torch.tensor(W_K_val, dtype=torch.float32),
        torch.tensor(W_V_val, dtype=torch.float32),
        torch.tensor(W_O_val, dtype=torch.float32),
        torch.tensor(W_1_val, dtype=torch.float32),
        torch.tensor(W_2_val, dtype=torch.float32),
        torch.tensor(b_1_val, dtype=torch.float32).reshape(1, 1, model_dim),
        torch.tensor(b_2_val, dtype=torch.float32).reshape(1, num_classes),
    ]

    # -------------------------
    # Define runner functions (SGD uses these)
    # -------------------------
    def f_run_model(X_batch: torch.Tensor, y_batch: torch.Tensor, model_weights: List[torch.Tensor]):
        W_Q_t, W_K_t, W_V_t, W_O_t, W_1_t, W_2_t, b_1_t, b_2_t = model_weights
        return evaluator.run(
            input_values={
                X_node: X_batch,
                y_node: y_batch,
                W_Q: W_Q_t,
                W_K: W_K_t,
                W_V: W_V_t,
                W_O: W_O_t,
                W_1: W_1_t,
                W_2: W_2_t,
                b_1: b_1_t,
                b_2: b_2_t,
            }
        )

    def f_eval_model(X_val: torch.Tensor, model_weights: List[torch.Tensor]):
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        all_logits = []

        W_Q_t, W_K_t, W_V_t, W_O_t, W_1_t, W_2_t, b_1_t, b_2_t = model_weights

        for i in range(num_batches):
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]

            logits = test_evaluator.run({
                X_node: X_batch,
                W_Q: W_Q_t,
                W_K: W_K_t,
                W_V: W_V_t,
                W_O: W_O_t,
                W_1: W_1_t,
                W_2: W_2_t,
                b_1: b_1_t,
                b_2: b_2_t,
            })
            all_logits.append(logits[0].detach().cpu().numpy())

        concatenated_logits = np.concatenate(all_logits, axis=0)
        return np.argmax(concatenated_logits, axis=1)

    # -------------------------
    # IMPORTANT: adjust SGD signature expectation
    # Your sgd_epoch must call: f_run_model(X_batch, y_batch, model_weights)
    # -------------------------
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)

        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        pred = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(pred == y_test_np)}, loss = {loss_val}"
        )

    pred = f_eval_model(X_test, model_weights)
    return np.mean(pred == y_test_np)


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
