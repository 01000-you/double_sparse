#!/bin/bash
# LLaMA Double Sparse Pruning - transformers 4.51.x 호환
# pip install transformers>=4.51.2

MODEL=${1:-"meta-llama/Llama-2-13b-hf"}
DATASET=${2:-"c4"}

for sp in 0.5 0.6 0.7; do
   for fm in "" "--fix-mask"; do
      for final in "" "--no-final"; do
        name="llama451-${sp}${fm}${final}"
        echo "Running: $name"
        python llama_451.py "$MODEL" "$DATASET" --sparsity $sp $fm $final | tee logs/${name}.log
      done
    done
done
