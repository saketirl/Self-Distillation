#!/bin/bash
# P-Gram ICL sweep: 3x3 (pgram_lr x adamw_lr), gram_lr pinned at 3e-4,
# full 20-task GQA protocol (matches GRAMFLOW_REPORT §7.5 references).
cd "$(dirname "$0")"
PY=/data/saket/continual/Self-Distillation/distillation/bin/python
mkdir -p eigenspectrum/outputs/pgram_sweep logs_pgram
i=0
for pl in 3e-4 1e-3 3e-3; do
  for al in 3e-4 1e-3 3e-3; do
    i=$((i+1))
    tag="pg${pl}_aw${al}"
    CUDA_VISIBLE_DEVICES=$i nohup $PY eigenspectrum/scripts/icl_pgram_sweep.py \
      --n_tasks 20 --n_kv_heads 1 --head_dim 48 \
      --configs pgram_gramflow \
      --pgram_lr $pl --adamw_lr $al --gram_lr 3e-4 \
      --out eigenspectrum/outputs/pgram_sweep/${tag}.npz \
      > logs_pgram/${tag}.log 2>&1 &
    echo "launched $tag on GPU $i (pid $!)"
  done
done
wait_pids=$(jobs -p)
echo "all 9 launched"
