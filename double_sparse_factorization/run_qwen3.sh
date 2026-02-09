#!/bin/bash
# Qwen3 Double Sparse Pruning 실행 스크립트
# transformers 4.51.2 이상 필요: pip install transformers>=4.51.2

MODEL=${1:-"Qwen/Qwen3-8B"}
DATASET=${2:-"c4"}

for sp in 0.5 0.6 0.7; do
   for fm in "" "--fix-mask"; do
      for final in "" "--no-final"; do
        name="qwen3-${sp}${fm}${final}"
        echo "Running: $name"
        python qwen3.py "$MODEL" "$DATASET" --sparsity $sp $fm $final | tee logs/${name}.log
      done
    done
done
