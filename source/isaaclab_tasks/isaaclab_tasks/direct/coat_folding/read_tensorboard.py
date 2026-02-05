import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import rcParams

# Optional: Make plots look nice
rcParams.update({
    "font.size": 12,
    "figure.figsize": (12, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})

def load_scalar_from_log(log_path, tag):
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise ValueError(f"Tag '{tag}' not found in {log_path}")
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

# === Your experiment log directories ===
log_dirs = [
    "/workspace/isaaclab/logs/rsl_rl/franka_cloth_direct/2025-07-04_15-29-58_initMotionIsZero",
    "/workspace/isaaclab/logs/rsl_rl/franka_cloth_direct/2025-07-05_11-59-45_oneUpdateNoRandomScale",
    "/workspace/isaaclab/logs/rsl_rl/franka_cloth_direct/2025-07-08_10-57-24_fastLearning"
]

# === Reward tags to compare ===
reward_tags = [
    "Episode/corner_x_reward",
    "Episode/direction_reward",
    "Episode/height_reward",
    "Episode/spread_reward"
]

# === Create subplot for each reward ===
fig, axes = plt.subplots(1, len(reward_tags), figsize=(4 * len(reward_tags), 4))

if len(reward_tags) == 1:
    axes = [axes]  # wrap in list for consistent indexing

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(log_dirs)))

window_size = 100  # adjust as needed

for i, tag in enumerate(reward_tags):
    ax = axes[i]
    for j, log_dir in enumerate(log_dirs):
        try:
            steps, values = load_scalar_from_log(log_dir, tag)
            # values = values[::1000]
            # steps = steps[::1000]

            # Compute rolling average
            if len(values) >= window_size:
                rolling = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                rolling_steps = steps[window_size - 1:]
            else:
                rolling = values
                rolling_steps = steps

            # Actual reward (transparent)
            ax.plot(steps, values, color=colors[j], alpha=0.25) # , label=f"Run {j+1} (raw)")

            # Rolling average (bold)
            ax.plot(rolling_steps, rolling, color=colors[j], label=f"Run {j+1}")

        except Exception as e:
            print(f"Error loading {tag} from {log_dir}: {e}")

    ax.set_title(tag.replace("Episode/", "").replace("_", " ").title(), fontsize=12)
    ax.set_xlabel("Step")
    if i == 0:
        ax.set_ylabel("Weighted reward")
    ax.legend()

# plt.suptitle("Comparison of Rewards Across Experiments", fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig("/workspace/isaaclab/logs/rsl_rl/franka_cloth_direct/reward_comparison.pdf")
