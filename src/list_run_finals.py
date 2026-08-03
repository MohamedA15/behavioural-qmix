import os
from plot_learning_curves import MODEL_FOLDERS, load_run

ROOT_BASE = None

def list_runs_for_model(model_folder):
    # locate folder relative to plot_learning_curves' ROOT
    from plot_learning_curves import ROOT
    folder = os.path.join(ROOT, model_folder)
    if not os.path.isdir(folder):
        print(f"Missing folder: {folder}")
        return
    for name in sorted(os.listdir(folder)):
        run_dir = os.path.join(folder, name)
        if not os.path.isdir(run_dir):
            continue
        loaded = load_run(run_dir)
        if loaded is None:
            print(f"{name}: no data")
            continue
        steps, returns = loaded
        last = returns[-1] if len(returns) else None
        print(f"{name}: final_return={last}")

def main():
    for label, folder in MODEL_FOLDERS.items():
        print(f"\n== {label} ({folder}) ==")
        list_runs_for_model(folder)

if __name__ == '__main__':
    main()
