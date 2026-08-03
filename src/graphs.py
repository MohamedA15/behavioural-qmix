import json
import numpy as np
import matplotlib.pyplot as plt

# Constraint A runs
RUNS = [
    r"results\sacred\8\info.json",
    r"results\sacred\9\info.json",
    r"results\sacred\10\info.json",
    r"results\sacred\11\info.json",
    r"results\sacred\12\info.json",
]

all_returns = []

for run in RUNS:
    with open(run, "r") as f:
        data = json.load(f)

    all_returns.append(data["test_return_mean"]["values"])
    steps = data["test_return_mean"]["steps"]

all_returns = np.array(all_returns)

mean = np.mean(all_returns, axis=0)
std = np.std(all_returns, axis=0)

plt.figure(figsize=(8,5))

plt.plot(
    steps,
    mean,
    linewidth=2.5,
    label="Constraint A"
)

plt.fill_between(
    steps,
    mean - std,
    mean + std,
    alpha=0.25
)

plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Mean Evaluation Return", fontsize=12)
plt.title("Constraint A Learning Curve", fontsize=13)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig("ConstraintA_LearningCurve.png", dpi=300)
plt.show()