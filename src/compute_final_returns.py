from plot_learning_curves import MODEL_FOLDERS, collect_model_runs
import numpy as np


def main():
    report = {}
    for label, folder in MODEL_FOLDERS.items():
        runs = collect_model_runs(folder)
        finals = []
        for steps, returns in runs:
            if len(returns) == 0:
                continue
            # take the last available evaluation point
            finals.append(float(returns[-1]))
        if len(finals) == 0:
            print(f"{label}: no runs or no returns")
            report[label] = None
            continue
        arr = np.array(finals)
        print(f"{label}: n_runs={len(finals)}, per-run finals={arr.tolist()}, mean={arr.mean():.3f}, std={arr.std():.3f}")
        report[label] = (arr.mean(), arr.std(), arr.tolist())

    # Print summary sorted by mean desc
    summary = [(k, v[0]) for k, v in report.items() if v is not None]
    summary.sort(key=lambda x: x[1], reverse=True)
    print('\nSummary (by final mean):')
    for k, m in summary:
        print(f" - {k}: {m:.3f}")


if __name__ == '__main__':
    main()
