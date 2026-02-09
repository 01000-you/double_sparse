#!/bin/bash
# Qwen3 2SSP + Double Sparse 결합 프루닝 실행 스크립트
# 프로세스: 1) AB 분해 2) 2SSP 채널 선택 3) ADMM 최적화
# MLP + Attention 모두 프루닝 (2SSP와 동일)
# transformers 4.51.2 이상 필요: pip install transformers>=4.51.2

MODEL=${1:-"Qwen/Qwen3-8B"}
DATASET=${2:-"c4"}

mkdir -p logs

# 옵션 1: target-compression (W 대비 A+B 저장 목표 압축률)
# 예: --target-compression 0.3 = 30% 압축 (저장=0.7*W)
echo "Running: qwen3-2ssp-ds (AB분해 -> 2SSP -> ADMM)"
python qwen3_2ssp.py "$MODEL" "$DATASET" \
  --target-compression 0.3 \
  --nsamples 128 \
  | tee logs/qwen3-2ssp-ds.log

# 옵션 2: channel-reduction (채널 직접 지정)
# python qwen3_2ssp.py "$MODEL" "$DATASET" --channel-reduction 0.5 --nsamples 128 | tee logs/qwen3-2ssp-ds.log

# MLP만 프루닝: --prune_only mlp
# 저장: --save ./pruned_model
