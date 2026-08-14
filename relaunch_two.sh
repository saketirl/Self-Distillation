#!/bin/bash
# Relaunch the two OOM'd sweep points (killed by a foreign 46GB process) on
# currently-free GPUs 0 and 3. expandable_segments guards against fragmentation
# if the neighbor comes back.
cd "$(dirname "$0")"
PY=/data/saket/continual/Self-Distillation/distillation/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 nohup $PY eigenspectrum/scripts/icl_pgram_sweep.py \
  --n_tasks 20 --n_kv_heads 1 --head_dim 48 --configs pgram_gramflow \
  --pgram_lr 3e-4 --adamw_lr 3e-3 --gram_lr 3e-4 \
  --out eigenspectrum/outputs/pgram_sweep/pg3e-4_aw3e-3.npz \
  > logs_pgram/pg3e-4_aw3e-3.log 2>&1 &
echo "relaunched pg3e-4_aw3e-3 on GPU 0 (pid $!)"
CUDA_VISIBLE_DEVICES=3 nohup $PY eigenspectrum/scripts/icl_pgram_sweep.py \
  --n_tasks 20 --n_kv_heads 1 --head_dim 48 --configs pgram_gramflow \
  --pgram_lr 1e-3 --adamw_lr 3e-3 --gram_lr 3e-4 \
  --out eigenspectrum/outputs/pgram_sweep/pg1e-3_aw3e-3.npz \
  > logs_pgram/pg1e-3_aw3e-3.log 2>&1 &
echo "relaunched pg1e-3_aw3e-3 on GPU 3 (pid $!)"
