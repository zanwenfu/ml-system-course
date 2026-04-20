# CSE 590 Programming Assignment 4 (Mixture of Experts)

Your task is to implement an Mixture of Expert (MoE) model with two different variants of communication patterns for the MoE layer: using tensor parallel (TP) and expert parallel (EP). 

As a refererence implementation, we provide the `SimpleMoE` class in the `moe.py` file. We provide the skeleton code for the `MoE_TP` and `MoE_EP` classes. You can test your implementation by running the `test_moe.py` file.

## Tensor Parallel (TP)

Every process holds all experts, but only a portion of each experts' weights. At expert forward pass, each process compute partial expert output, and then perform all-reduce to get the full expert output.

- Implement `ShardedLinear` class in the `moe.py` file.
- Implement `MoE_TP` class in the `moe.py` file.

## Expert Parallel (EP)

Each process holds a subset of experts in its entirety. At expert forward pass, each process compute the output for its assigned slice of the input, and then perform all-to-all communication to get the full expert output.

- Implement `MoE_EP` class in the `moe.py` file.


## Benchmark your implementation
- Answer the two questions in `analysis.md` about your implementation. 

## How to Submit Your Homework

In your programming assignment root directory run
```bash
make handin.zip
```
Then you will see a `handin.zip` file under your root directory, please go to Gradescope and submit the zip file.


