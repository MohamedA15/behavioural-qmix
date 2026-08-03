import itertools
from plot_learning_curves import MODEL_FOLDERS, collect_model_runs
import numpy as np

# Target means from the validated table
TARGETS = {
    "Baseline": 0.672,
    "A": 0.784,
    "B": 0.730,
    "C": 0.874,
    "D": 0.628,
}

TOL = 0.005  # tolerance for matching mean


def find_subsets(values, target, k=5, tol=TOL):
    # values: list of floats
    idxs = list(range(len(values)))
    for combo in itertools.combinations(idxs, k):
        s = np.mean([values[i] for i in combo])
        if abs(s - target) <= tol:
            return combo, s
    return None, None


def main():
    for label, folder in MODEL_FOLDERS.items():
        runs = collect_model_runs(folder)
        finals = [float(returns[-1]) for steps, returns in runs if len(returns)]
        print(f"\n== {label} finals (n={len(finals)}) ==")
        print(finals)
        target = TARGETS.get(label)
        if target is None:
            continue
        combo, mean_val = find_subsets(finals, target)
        if combo is not None:
            print(f"Found subset indices {combo} with mean {mean_val:.3f}")
            print("Values:", [finals[i] for i in combo])
        else:
            print(f"No subset of 5 runs matches target {target} within tol={TOL}")


if __name__ == '__main__':
    main()
