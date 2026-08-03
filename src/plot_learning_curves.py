import json
import os
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "sacred", "qmix"))

# Map label -> folder name under ROOT
MODEL_FOLDERS = {
    "Baseline": "lbforaging_Foraging-6x6-2p-1f-v3",
    "A": "lbforaging_Foraging-6x6-2p-1f-v3_ConstraintA",
    "B": "lbforaging_Foraging-6x6-2p-1f-v3_ConstraintB",
    "C": "lbforaging_Foraging-6x6-2p-1f-v3_ConstraintC",
    "D": "lbforaging_Foraging-6x6-2p-1f-v3_ConstraintD",
}

# How labels should appear in the legend (more descriptive)
DISPLAY_NAMES = {
    "Baseline": "Baseline",
    "A": "Constraint A",
    "B": "Constraint B",
    "C": "Constraint C",
    "D": "Constraint D",
}


def load_run(run_path):
    """Load a single run directory and return (steps, returns) arrays.
    Supports both info.json (flat arrays with _T suffix) and metrics.json (dict with steps/values).
    Returns None if no usable file found.
    """
    info_path = os.path.join(run_path, "info.json")
    metrics_path = os.path.join(run_path, "metrics.json")

    if os.path.isfile(info_path):
        with open(info_path, "r") as f:
            data = json.load(f)
        # info.json format: keys like 'test_return_mean' and 'test_return_mean_T'
        if "test_return_mean" in data:
            raw_returns = data["test_return_mean"]
            # sometimes entries are dicts (with value/timestamp). Extract numeric values if needed.
            def extract_scalar(x):
                if isinstance(x, (int, float)):
                    return x
                if isinstance(x, dict):
                    for v in x.values():
                        if isinstance(v, (int, float)):
                            return v
                return None

            returns_list = []
            for item in raw_returns:
                val = extract_scalar(item)
                if val is None:
                    # fallback: try converting directly
                    try:
                        val = float(item)
                    except Exception:
                        val = np.nan
                returns_list.append(val)
            returns = np.array(returns_list)
            steps_key = "test_return_mean_T"
            steps = np.array(data.get(steps_key) or data.get("test_return_mean_T")) if steps_key in data or "test_return_mean_T" in data else None
            if steps is None:
                # try generic 'test_return_mean_T' name fallback
                for k in data.keys():
                    if k.lower().endswith("_t"):
                        steps = np.array(data[k])
                        break
            if steps is None:
                return None
            return steps.astype(float), returns.astype(float)

    if os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        # metrics.json format: 'test_return_mean': { 'steps': [...], 'values': [...] }
        tr = data.get("test_return_mean")
        if isinstance(tr, dict) and "steps" in tr and ("values" in tr or "vals" in tr):
            steps = np.array(tr["steps"]).astype(float)
            vals = tr.get("values") or tr.get("vals")
            returns = np.array(vals).astype(float)
            return steps, returns

    return None


def collect_model_runs(model_folder):
    folder = os.path.join(ROOT, model_folder)
    if not os.path.isdir(folder):
        print(f"Warning: folder not found: {folder}")
        return []

    runs = []
    for name in sorted(os.listdir(folder), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
        run_dir = os.path.join(folder, name)
        if not os.path.isdir(run_dir):
            continue
        loaded = load_run(run_dir)
        if loaded is None:
            # try deeper (some experiments keep info.json under a subdir)
            info_candidate = os.path.join(run_dir, "info.json")
            if os.path.isfile(info_candidate):
                loaded = load_run(run_dir)
        if loaded is not None:
            runs.append(loaded)
    return runs


def make_common_grid(all_runs, num_points=500):
    max_step = 0
    for runs in all_runs:
        for steps, _ in runs:
            if len(steps) > 0:
                max_step = max(max_step, steps.max())
    if max_step <= 0:
        return None
    return np.linspace(0, max_step, num=num_points)


def aggregate_runs(runs, grid):
    # runs: list of (steps, returns)
    interp_vals = []
    for steps, returns in runs:
        # ensure monotonic steps
        order = np.argsort(steps)
        s = np.array(steps)[order]
        r = np.array(returns)[order]
        # fill and extrapolate with edge values
        v = np.interp(grid, s, r, left=r[0], right=r[-1])
        interp_vals.append(v)
    if len(interp_vals) == 0:
        return None, None
    arr = np.vstack(interp_vals)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_learning_curves(output_path="figure1_learning_curve.png"):
    all_model_runs = {}
    for label, folder in MODEL_FOLDERS.items():
        runs = collect_model_runs(folder)
        print(f"Found {len(runs)} runs for {label}")
        all_model_runs[label] = runs

    grid = make_common_grid(list(all_model_runs.values()), num_points=600)
    if grid is None:
        raise RuntimeError("No steps found in any run to build a grid.")


    plt.figure(figsize=(9, 5))

    colors = {
        "Baseline": "tab:gray",
        "A": "tab:blue",
        "B": "tab:orange",
        "C": "tab:green",
        "D": "tab:red",
    }

    for label in ["Baseline", "A", "B", "C", "D"]:
        runs = all_model_runs.get(label, [])
        mean, std = aggregate_runs(runs, grid)
        if mean is None:
            print(f"Skipping {label}: no runs")
            continue
        disp = DISPLAY_NAMES.get(label, label)
        plt.plot(grid, mean, label=disp, linewidth=2.5, color=colors.get(label))
        plt.fill_between(grid, mean - std, mean + std, alpha=0.15, color=colors.get(label))

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Mean Evaluation Return", fontsize=12)
    plt.title("Learning Performance on the 6×6 Training Environment", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()

    # Limit y-axis for cleaner presentation (returns are in [0, 1])
    plt.ylim(0, 1.0)

    # Improve x-axis ticks to show nice round numbers (0, 100k, 200k...)
    max_step = int(grid.max())
    step_tick = max(1, int(round(max_step / 5 / 1000.0)) * 1000)
    ticks = np.linspace(0, max_step, 6, dtype=int)
    plt.xticks(ticks)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved learning curve to {output_path}")


if __name__ == "__main__":
    plot_learning_curves()
