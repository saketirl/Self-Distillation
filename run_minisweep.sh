#!/bin/bash
# Joint mini-sweep: gram_lr x pgram_lr x adamw_lr, 8 points, 4 GPUs (1,2,3,4),
# two waves. Crosses the gram axis with both frontier pg values + joint moves.
cd "$(dirname "$0")"
PY=/data/saket/continual/Self-Distillation/distillation/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=(1 2 3 4)
POINTS=(
  "1e-3 3e-4 1e-4"   # pg aw gram : gram axis at chain winner
  "1e-3 3e-4 1e-3"
  "3e-3 3e-4 1e-4"   # gram axis at balanced point
  "3e-3 3e-4 1e-3"
  "1e-3 1e-4 1e-3"   # joint: low aw + high gram
  "3e-3 1e-4 1e-4"   # joint: low aw + low gram
  "1e-3 1e-3 1e-3"   # all-mid joint move
  "3e-3 3e-4 3e-3"   # strong gram at balanced point
)
launch () {
  local idx=$1 pg=$2 aw=$3 gl=$4
  local tag="pg${pg}_aw${aw}_gl${gl}"
  CUDA_VISIBLE_DEVICES=${GPUS[$idx]} nohup $PY eigenspectrum/scripts/icl_pgram_sweep.py \
    --n_tasks 20 --n_kv_heads 1 --head_dim 48 --configs pgram_gramflow \
    --pgram_lr $pg --adamw_lr $aw --gram_lr $gl \
    --out eigenspectrum/outputs/pgram_sweep/${tag}.npz \
    > logs_pgram/${tag}.log 2>&1 &
  echo "launched ${tag} on GPU ${GPUS[$idx]} (pid $!)"
}
for i in 0 1 2 3; do set -- ${POINTS[$i]}; launch $i $1 $2 $3; done
wait
echo "wave 1 done"
for i in 4 5 6 7; do set -- ${POINTS[$i]}; launch $((i-4)) $1 $2 $3; done
wait
echo "MINISWEEP DONE"
