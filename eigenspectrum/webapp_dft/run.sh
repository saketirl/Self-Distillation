#!/bin/bash
# Run the Base vs DFT e06 eigenspectrum viewer
# Defaults to eigenspectrum/outputs/qwen3/ relative to this script's grandparent dir
# Override with: EIGENSPECTRUM_OUTPUT_DIR=/path/to/outputs/qwen3 bash run.sh

cd "$(dirname "$0")"
pip install flask numpy -q
python app.py
