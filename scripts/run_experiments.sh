#!/bin/bash
# Run sequential continual learning experiments
# Compares SDFT vs SFT on 3 tasks: Tool Selection -> JSON Extraction -> Text-to-SQL

set -e

# Configuration
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_BASE="outputs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Training settings
LEARNING_RATE=2e-5
NUM_EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=16

# Optional: limit samples for quick testing
# MAX_SAMPLES="--max_samples_per_task 100"
MAX_SAMPLES=""

echo "=============================================="
echo "Sequential Continual Learning Experiments"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Timestamp: $TIMESTAMP"
echo ""

# Run SDFT experiment
echo "Running SDFT experiment..."
python train_sequential.py \
    --method sdft \
    --model_name $MODEL_NAME \
    --output_dir "${OUTPUT_BASE}/sdft_${TIMESTAMP}" \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    $MAX_SAMPLES

echo ""
echo "SDFT experiment complete!"
echo ""

# Run SFT experiment
echo "Running SFT experiment..."
python train_sequential.py \
    --method sft \
    --model_name $MODEL_NAME \
    --output_dir "${OUTPUT_BASE}/sft_${TIMESTAMP}" \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    $MAX_SAMPLES

echo ""
echo "SFT experiment complete!"
echo ""

# Compare results
echo "=============================================="
echo "Comparing Results"
echo "=============================================="
python -c "
import json

sdft_results = json.load(open('${OUTPUT_BASE}/sdft_${TIMESTAMP}/results.json'))
sft_results = json.load(open('${OUTPUT_BASE}/sft_${TIMESTAMP}/results.json'))

print('Task Order:', sdft_results['tasks'])
print()

tasks = sdft_results['tasks']
for method, results in [('SDFT', sdft_results), ('SFT', sft_results)]:
    print(f'{method} Results:')
    final = results['results_per_stage'][-1]['results']
    for task in tasks:
        acc = final[task]['accuracy']
        print(f'  {task}: {acc:.2%}')
    print()

print('Forgetting Comparison:')
for task_idx, task in enumerate(tasks[:-1]):
    sdft_after = sdft_results['results_per_stage'][task_idx]['results'][task]['accuracy']
    sdft_final = sdft_results['results_per_stage'][-1]['results'][task]['accuracy']
    sdft_forgetting = sdft_after - sdft_final

    sft_after = sft_results['results_per_stage'][task_idx]['results'][task]['accuracy']
    sft_final = sft_results['results_per_stage'][-1]['results'][task]['accuracy']
    sft_forgetting = sft_after - sft_final

    print(f'  {task}:')
    print(f'    SDFT: {sdft_forgetting:.2%} forgetting')
    print(f'    SFT:  {sft_forgetting:.2%} forgetting')
"

echo ""
echo "All experiments complete!"
echo "Results saved to: ${OUTPUT_BASE}/sdft_${TIMESTAMP} and ${OUTPUT_BASE}/sft_${TIMESTAMP}"
