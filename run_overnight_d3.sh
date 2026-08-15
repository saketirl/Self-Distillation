#!/bin/bash
# Overnight Dion3-upgrade ablation — SEQUENTIAL on GPU 8 ONLY.
# All configs at the bracketed ICL winner (pg3e-3 / aw1e-4 / gl1e-4, r8)
# unless stated. ~10 x ~40min = ~7h. References: pgram(per-head) 11.44 (r8),
# 11.34 (r16); compdual_gramflow 11.44; adam 28.11.
cd "$(dirname "$0")"
PY=/data/saket/continual/Self-Distillation/distillation/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=8
W="--n_tasks 20 --n_kv_heads 1 --head_dim 48 --pgram_lr 3e-3 --adamw_lr 1e-4 --gram_lr 1e-4"
OUT=eigenspectrum/outputs/pgram_d3
mkdir -p $OUT logs_pgram
run () {
  local tag=$1; shift
  echo "=== $(date +%H:%M) launching $tag"
  $PY eigenspectrum/scripts/icl_pgram_sweep.py $W \
    --out $OUT/${tag}.npz "$@" > logs_pgram/d3_${tag}.log 2>&1
  grep -h "done=" logs_pgram/d3_${tag}.log | tail -1
}
# 1. pooled control (all features off) — isolates pooled-vs-perhead delta
run pooled_base      --configs pgram_pooled_gramflow
# 2-4. single features
run mom_only         --configs pgramd3_gramflow --d3_normuon 0 --d3_gramns 0
run normuon_only     --configs pgramd3_gramflow --d3_momentum 0 --d3_gramns 0
run gramns_only      --configs pgramd3_gramflow --d3_momentum 0 --d3_normuon 0
# 5. mom + normuon (the loss-gap pair)
run mom_normuon      --configs pgramd3_gramflow --d3_gramns 0
# 6. all three
run all3             --configs pgramd3_gramflow
# 7. all three at the chain-retention lr point
run all3_lrB         --configs pgramd3_gramflow --pgram_lr 1e-3 --adamw_lr 3e-4 --gram_lr 1e-3
# 8. momentum 0.95 variant of the pair
run mom95_normuon    --configs pgramd3_gramflow --d3_gramns 0 --d3_momentum 0.95
# 9. all3 at r16 (nominal r-optimum)
run all3_r16         --configs pgramd3_gramflow --pgram_rank 16
# 10. adam control rerun for same-session sanity
run adam_ctl         --configs adam
echo "OVERNIGHT_D3_DONE $(date)"
