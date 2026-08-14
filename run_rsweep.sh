#!/bin/bash
# Final stage: (a) outward brackets for the winner's two boundary axes,
# (b) r-sweep {2,4,16,32} at the winner pg3e-3/aw1e-4/gl1e-4 (r=8 = the
# winner itself, already measured). 4-wide, two waves.
cd "$(dirname "$0")"
PY=/data/saket/continual/Self-Distillation/distillation/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=(1 2 3 4)
run () {
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY eigenspectrum/scripts/icl_pgram_sweep.py \
    --n_tasks 20 --n_kv_heads 1 --head_dim 48 --configs pgram_gramflow "$@" \
    --out eigenspectrum/outputs/pgram_sweep/${tag}.npz \
    > logs_pgram/${tag}.log 2>&1 &
  echo "launched ${tag} on GPU ${gpu} (pid $!)"
}
W="--adamw_lr 1e-4 --gram_lr 1e-4"
# wave A: brackets + small-r
run 1 bracket_pg1e-2_aw1e-4_gl1e-4 --pgram_lr 1e-2 $W
run 2 bracket_pg3e-3_aw3e-5_gl1e-4 --pgram_lr 3e-3 --adamw_lr 3e-5 --gram_lr 1e-4
run 3 rsweep_r2  --pgram_lr 3e-3 $W --pgram_rank 2
run 4 rsweep_r4  --pgram_lr 3e-3 $W --pgram_rank 4
wait
echo "wave A done"
run 1 rsweep_r16 --pgram_lr 3e-3 $W --pgram_rank 16
run 2 rsweep_r32 --pgram_lr 3e-3 $W --pgram_rank 32
wait
echo "RSWEEP DONE"
