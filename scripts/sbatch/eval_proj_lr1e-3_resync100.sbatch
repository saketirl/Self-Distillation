#!/bin/bash
#SBATCH --job-name=eval_proj_lr1e-3
#SBATCH --output=logs/eval_proj_lr1e-3_%j.out
#SBATCH --error=logs/eval_proj_lr1e-3_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Change to working directory
cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

# Set environment variables
export HF_HOME="/home/guests/saket/models"

MODEL_PATH="/home/guests/saket/continual/Self-Distillation/outputs/projected_proj_lr1e-3_resync100"
MAX_SAMPLES=100

echo "========================================"
echo "Evaluating: $MODEL_PATH"
echo "Max samples: $MAX_SAMPLES"
echo "========================================"

uv run python << EOF
import sys
sys.path.insert(0, '.')

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_loaders import load_task_dataset
from scripts.eval_projected_models import evaluate_tooluse

# Load model
model_path = "${MODEL_PATH}"
max_samples = ${MAX_SAMPLES}
print(f'Loading model from {model_path}...')

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load eval dataset
print('Loading tooluse evaluation dataset...')
_, eval_dataset = load_task_dataset('tooluse', seed=42)
print(f'Eval dataset size: {len(eval_dataset)}')

# Evaluate
print(f'Evaluating on {max_samples} samples...')
results = evaluate_tooluse(
    model, tokenizer, eval_dataset,
    device='cuda',
    max_samples=max_samples,
)

print()
print('=' * 60)
print('RESULTS')
print('=' * 60)
print(f'Model: {model_path}')
print(f'Accuracy: {results["accuracy"]:.2%} ({results["correct"]}/{results["total"]})')
print()

# Show some examples
print('Sample predictions:')
for i, r in enumerate(results['results'][:5]):
    print(f'  [{i+1}] Correct: {r["correct"]}')
    print(f'      Expected: {r["expected"][:100]}...')
    print(f'      Generated: {r["generated"][:100]}...')
    print()

# Save results to JSON
output_file = Path(model_path) / 'eval_results.json'
eval_data = {
    'timestamp': datetime.now().isoformat(),
    'model_path': model_path,
    'task': 'tooluse',
    'max_samples': max_samples,
    'accuracy': results['accuracy'],
    'correct': results['correct'],
    'total': results['total'],
    'results': results['results'],
}
with open(output_file, 'w') as f:
    json.dump(eval_data, f, indent=2)
print(f'Results saved to: {output_file}')
EOF

echo "Done!"
