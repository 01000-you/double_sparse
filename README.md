# Double sparse pruning

Code for the ICLR 2025 paper: [Two Sparse Matrices are Better than One: Sparsifying Neural Networks with Double Sparse Factorization](https://openreview.net/forum?id=DwiwOcK1B7).

The repository is based on [SparseGPT](https://github.com/IST-DASLab/sparsegpt) code 

## Dependencies

* `torch`: tested on v2.2.1
* `transformers`: tested on v4.35.2
* `datasets`: tested on v2.16.1

## Usage

We also provide LLaMA pruning script with the very same interface:

```
# Sparsify LLaMa with DSF
python llama.py meta-llama/Llama-2-7b-hf c4 --sparsity 0.5
```

## Other experiments

For replicating other experiments (comparision with OBC a post-training pruning with finetuning)
see `other_experiments` directory.

### Kernels

[Kernels for DSF by Elvir Crnčević ](https://github.com/elvircrn/double_sparse_kernel)
