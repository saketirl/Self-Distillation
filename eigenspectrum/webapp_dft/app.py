"""Flask web application for comparing Base vs DFT e06 eigenspectrums."""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

BASE_DIR = Path(os.environ.get(
    "EIGENSPECTRUM_OUTPUT_DIR",
    str(Path(__file__).parent.parent / "outputs" / "qwen3")
))

print(f"Output directory: {BASE_DIR}")

CHECKPOINTS = {
    "base": "base",
    "task_0": "dft_e06/task_0_tooluse",
    "task_1": "dft_e06/task_1_cot_math",
    "task_2": "dft_e06/task_2_spider",
}

DFT_TASK_NAMES = {
    "task_0": "Task 0: ToolUse",
    "task_1": "Task 1: CoT Math",
    "task_2": "Task 2: Spider",
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


def get_layer_dir(checkpoint_key, layer):
    subdir = CHECKPOINTS[checkpoint_key]
    return BASE_DIR / subdir / f"layer_{layer}"


def get_available_layers(checkpoint_key):
    subdir = CHECKPOINTS[checkpoint_key]
    ckpt_dir = BASE_DIR / subdir
    if not ckpt_dir.exists():
        return []
    layers = []
    for layer_dir in ckpt_dir.glob("layer_*"):
        if layer_dir.is_dir():
            try:
                layers.append(int(layer_dir.name.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return sorted(layers)


def load_spectrum_data(checkpoint_key, layer, param):
    try:
        layer_dir = get_layer_dir(checkpoint_key, layer)
        if not layer_dir.exists():
            return None

        safe_param = param.replace(".", "_")
        npz_files = list(layer_dir.glob(f"*{safe_param}*_spectrum.npz"))
        if not npz_files:
            return None

        data = np.load(str(npz_files[0]), allow_pickle=True)

        eigenvalues = []
        if "eigenvalues" in data:
            eig = data["eigenvalues"]
            eigenvalues = eig.flatten().tolist() if eig.ndim > 1 else eig.tolist()

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
        print(f"Error loading {checkpoint_key}/layer_{layer}/{param}: {e}")
        return None


def load_summary(checkpoint_key, layer):
    summary_path = get_layer_dir(checkpoint_key, layer) / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        with open(summary_path) as f:
            return json.load(f)
    except Exception:
        return {}


@app.route("/")
def index():
    return render_template("index.html", dft_tasks=DFT_TASK_NAMES, params=PARAM_NAMES)


@app.route("/api/layers")
def api_layers():
    dft_task = request.args.get("dft_task", "task_0")
    base_layers = get_available_layers("base")
    dft_layers = get_available_layers(dft_task)
    common = sorted(set(base_layers) & set(dft_layers))
    return jsonify({
        "base_layers": base_layers,
        "dft_layers": dft_layers,
        "common_layers": common if common else base_layers or dft_layers,
    })


@app.route("/api/spectrum")
def api_spectrum():
    dft_task = request.args.get("dft_task", "task_0")
    layer = int(request.args.get("layer", 0))
    param = request.args.get("param", "self_attn.q_proj.weight")

    return jsonify({
        "base": load_spectrum_data("base", layer, param),
        "dft": load_spectrum_data(dft_task, layer, param),
        "dft_task": dft_task,
        "layer": layer,
        "param": param,
    })


@app.route("/api/summary")
def api_summary():
    dft_task = request.args.get("dft_task", "task_0")
    layer = int(request.args.get("layer", 0))
    return jsonify({
        "base": load_summary("base", layer),
        "dft": load_summary(dft_task, layer),
    })


@app.route("/api/cross_layer")
def api_cross_layer():
    dft_task = request.args.get("dft_task", "task_0")
    param = request.args.get("param", "self_attn.q_proj.weight")

    base_data = {"layers": [], "top_eig": [], "eff_rank": [], "trace": []}
    dft_data = {"layers": [], "top_eig": [], "eff_rank": [], "trace": []}

    for layer in range(36):
        base_sp = load_spectrum_data("base", layer, param)
        if base_sp:
            base_data["layers"].append(layer)
            base_data["top_eig"].append(base_sp["top_eigenvalue"])
            base_data["eff_rank"].append(base_sp["effective_rank"])
            base_data["trace"].append(base_sp["trace_estimate"])

        dft_sp = load_spectrum_data(dft_task, layer, param)
        if dft_sp:
            dft_data["layers"].append(layer)
            dft_data["top_eig"].append(dft_sp["top_eigenvalue"])
            dft_data["eff_rank"].append(dft_sp["effective_rank"])
            dft_data["trace"].append(dft_sp["trace_estimate"])

    return jsonify({"base": base_data, "dft": dft_data})


if __name__ == "__main__":
    print("Starting server at http://localhost:5001")
    app.run(debug=True, port=5001, threaded=False)
