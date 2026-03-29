"""Flask web application for viewing eigenspectrum results side by side."""

import os
import json
import traceback
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

# Base directory for eigenspectrum outputs
BASE_DIR = Path(os.environ.get(
    "EIGENSPECTRUM_OUTPUT_DIR",
    str(Path(__file__).parent.parent / "outputs")
))

print(f"Output directory: {BASE_DIR}")

# Model configurations
MODELS = {
    "sft": {
        "base": "qwen2.5-3b-instruct",
        "task_0": "sft_full_mbpp/task_0_tooluse",
        "task_1": "sft_full_mbpp/task_1_gsm8k",
        "task_2": "sft_full_mbpp/task_2_mbpp",
    },
    "sdft": {
        "base": "qwen2.5-3b-instruct_sdft",
        "task_0": "sdft_full_mbpp/task_0_tooluse",
        "task_1": "sdft_full_mbpp/task_1_gsm8k",
        "task_2": "sdft_full_mbpp/task_2_mbpp",
    },
}

TASK_NAMES = {
    "base": "Base Model",
    "task_0": "Task 0: ToolUse",
    "task_1": "Task 1: GSM8K",
    "task_2": "Task 2: MBPP",
}

PARAM_NAMES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def get_checkpoint_dir(model_type, task):
    """Get the checkpoint directory for a model type and task."""
    model_subdir = MODELS[model_type][task]
    model_dir = BASE_DIR / model_subdir

    if model_dir.exists():
        checkpoints = list(model_dir.glob("checkpoint-*"))
        if checkpoints:
            return checkpoints[0]
    return model_dir


def get_available_layers(model_type, task):
    """Get list of available layers for a model."""
    checkpoint_dir = get_checkpoint_dir(model_type, task)
    if not checkpoint_dir.exists():
        return []

    layers = []
    for layer_dir in checkpoint_dir.glob("layer_*"):
        if layer_dir.is_dir():
            try:
                layer_idx = int(layer_dir.name.split("_")[1])
                layers.append(layer_idx)
            except (ValueError, IndexError):
                pass
    return sorted(layers)


def load_spectrum_data(model_type, task, layer, param):
    """Load spectrum data for a specific parameter."""
    try:
        checkpoint_dir = get_checkpoint_dir(model_type, task)
        layer_dir = checkpoint_dir / f"layer_{layer}"

        if not layer_dir.exists():
            return None

        safe_param = param.replace(".", "_")
        npz_files = list(layer_dir.glob(f"*{safe_param}*_spectrum.npz"))

        if not npz_files:
            return None

        npz_path = npz_files[0]
        data = np.load(str(npz_path), allow_pickle=True)

        # Flatten eigenvalues from all Lanczos samples
        eigenvalues = []
        if "eigenvalues" in data:
            eig = data["eigenvalues"]
            if eig.ndim > 1:
                eigenvalues = eig.flatten().tolist()
            else:
                eigenvalues = eig.tolist()

        result = {
            "density": data["density"].tolist(),
            "grids": data["grids"].tolist(),
            "eigenvalues": eigenvalues,
            "effective_rank": float(data["effective_rank"]),
            "top_eigenvalue": float(data["top_eigenvalue"]),
            "trace_estimate": float(data["trace_estimate"]),
            "num_params": int(data["num_params"]) if "num_params" in data else 0,
        }
        data.close()
        return result
    except Exception as e:
        print(f"Error loading {model_type}/{task}/layer_{layer}/{param}: {e}")
        return None


def load_summary(model_type, task, layer):
    """Load summary.json for a layer."""
    checkpoint_dir = get_checkpoint_dir(model_type, task)
    summary_path = checkpoint_dir / f"layer_{layer}" / "summary.json"

    if not summary_path.exists():
        return {}

    try:
        with open(summary_path) as f:
            return json.load(f)
    except Exception:
        return {}


@app.route("/")
def index():
    return render_template("index.html", tasks=TASK_NAMES, params=PARAM_NAMES)


@app.route("/api/layers")
def api_layers():
    task = request.args.get("task", "base")
    sft_layers = get_available_layers("sft", task)
    sdft_layers = get_available_layers("sdft", task)
    common_layers = sorted(set(sft_layers) & set(sdft_layers))

    return jsonify({
        "sft_layers": sft_layers,
        "sdft_layers": sdft_layers,
        "common_layers": common_layers if common_layers else sft_layers or sdft_layers,
    })


@app.route("/api/spectrum")
def api_spectrum():
    task = request.args.get("task", "base")
    layer = int(request.args.get("layer", 0))
    param = request.args.get("param", "self_attn.q_proj.weight")

    sft_data = load_spectrum_data("sft", task, layer, param)
    sdft_data = load_spectrum_data("sdft", task, layer, param)

    return jsonify({
        "sft": sft_data,
        "sdft": sdft_data,
        "task": task,
        "layer": layer,
        "param": param,
    })


@app.route("/api/summary")
def api_summary():
    task = request.args.get("task", "base")
    layer = int(request.args.get("layer", 0))

    return jsonify({
        "sft": load_summary("sft", task, layer),
        "sdft": load_summary("sdft", task, layer),
    })


@app.route("/api/cross_layer")
def api_cross_layer():
    task = request.args.get("task", "base")
    param = request.args.get("param", "self_attn.q_proj.weight")

    sft_data = {"layers": [], "top_eig": [], "eff_rank": [], "trace": []}
    sdft_data = {"layers": [], "top_eig": [], "eff_rank": [], "trace": []}

    for layer in range(36):
        sft_spectrum = load_spectrum_data("sft", task, layer, param)
        if sft_spectrum:
            sft_data["layers"].append(layer)
            sft_data["top_eig"].append(sft_spectrum["top_eigenvalue"])
            sft_data["eff_rank"].append(sft_spectrum["effective_rank"])
            sft_data["trace"].append(sft_spectrum["trace_estimate"])

        sdft_spectrum = load_spectrum_data("sdft", task, layer, param)
        if sdft_spectrum:
            sdft_data["layers"].append(layer)
            sdft_data["top_eig"].append(sdft_spectrum["top_eigenvalue"])
            sdft_data["eff_rank"].append(sdft_spectrum["effective_rank"])
            sdft_data["trace"].append(sdft_spectrum["trace_estimate"])

    return jsonify({"sft": sft_data, "sdft": sdft_data})


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(debug=True, port=5000, threaded=False)
