import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import math

def load_csv(path):
    data = []
    try:
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                d = {}
                for k, v in row.items():
                    try:
                        d[k] = int(v)
                    except ValueError:
                        try:
                            d[k] = float(v)
                        except ValueError:
                            d[k] = v
                data.append(d)
    except FileNotFoundError:
        print(f"Warning: File {path} not found.")
    return data

def aggregate_data(data):
    if not data:
        return []

    # Group by episode
    episodes = defaultdict(lambda: defaultdict(list))
    for row in data:
        ep = row.get('episode')
        if ep is None:
            continue
        for k, v in row.items():
            if k not in ['episode', 'seed'] and isinstance(v, (int, float)):
                episodes[ep][k].append(v)

    # Calculate means and standard deviations across seeds
    aggregated = []
    for ep in sorted(episodes.keys()):
        agg_row = {'episode': ep}
        for k, vals in episodes[ep].items():
            if vals:
                mean = sum(vals) / len(vals)
                agg_row[k] = mean

                # Calculate standard deviation
                if len(vals) > 1:
                    variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                    agg_row[f"{k}_std"] = math.sqrt(variance)
                else:
                    agg_row[f"{k}_std"] = 0.0

        aggregated.append(agg_row)

    return aggregated

def load_and_aggregate(path):
    return aggregate_data(load_csv(path))

def plot_smooth(x, y, y_std, label, color, window=50):
    if not x or not y:
        return

    # Smooth the mean
    smoothed_y = [sum(y[max(0, i-window+1):i+1]) / len(y[max(0, i-window+1):i+1]) for i in range(len(y))]

    # Smooth the standard deviation (optional, but makes it cleaner)
    smoothed_std = [sum(y_std[max(0, i-window+1):i+1]) / len(y_std[max(0, i-window+1):i+1]) for i in range(len(y_std))]

    # Calculate upper and lower bounds for the shaded area
    upper_bound = [my + mstd for my, mstd in zip(smoothed_y, smoothed_std)]
    lower_bound = [my - mstd for my, mstd in zip(smoothed_y, smoothed_std)]

    # Plot the smoothed mean
    plt.plot(x, smoothed_y, label=label, color=color, linewidth=1.5)

    # Fill the area between the upper and lower bounds with the standard deviation
    plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2)

reinforce_small_data = load_and_aggregate('../logs/reinforce_small.csv')
reinforce_big_data = load_and_aggregate('../logs/reinforce_big.csv')
actorcritic_oldcourse_data = load_and_aggregate('../logs/actorcritic_logs_oldcourse.csv')
actorcritic_data = load_and_aggregate('../logs/actorcritic.csv')
actorcritic_batched_data = load_and_aggregate('../logs/actorcritic_batched.csv')
actorcritic_randomized_data = load_and_aggregate('../logs/actorcritic_randomized.csv')
actorcritic_2buoy = load_and_aggregate('../logs/actorcritic_2buoy.csv')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plot_smooth([d["episode"] for d in reinforce_big_data], [d.get("total_reward", 0) for d in reinforce_big_data], [d.get("total_reward_std", 0) for d in reinforce_big_data], "Reinforce Big", 'orange')
plot_smooth([d["episode"] for d in reinforce_small_data], [d.get("total_reward", 0) for d in reinforce_small_data], [d.get("total_reward_std", 0) for d in reinforce_small_data], "Reinforce Small", 'blue')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Ticks by Episode")
plot_smooth([d["episode"] for d in reinforce_big_data], [d.get("ticks", 0) for d in reinforce_big_data], [d.get("ticks_std", 0) for d in reinforce_big_data], "Reinforce Big", 'orange')
plot_smooth([d["episode"] for d in reinforce_small_data], [d.get("ticks", 0) for d in reinforce_small_data], [d.get("ticks_std", 0) for d in reinforce_small_data], "Reinforce Small", 'blue')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reinforce_comparison.png")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Ticks by Episode")
plot_smooth([d["episode"] for d in reinforce_big_data], [d.get("ticks", 0) for d in reinforce_big_data], [d.get("ticks_std", 0) for d in reinforce_big_data], "Reinforce Big", 'orange')
plot_smooth([d["episode"] for d in actorcritic_oldcourse_data], [d.get("ticks", 0) for d in actorcritic_oldcourse_data], [d.get("ticks_std", 0) for d in actorcritic_oldcourse_data], "Actor-Critic Old Course", 'green')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Buoys Passed by Episode")
plot_smooth([d["episode"] for d in reinforce_small_data], [d.get("buoys_passed", 0) for d in reinforce_small_data], [d.get("buoys_passed_std", 0) for d in reinforce_small_data], "Reinforce Small", 'blue')
plot_smooth([d["episode"] for d in actorcritic_oldcourse_data], [d.get("buoys_passed", 0) for d in actorcritic_oldcourse_data], [d.get("buoys_passed_std", 0) for d in actorcritic_oldcourse_data], "Actor-Critic Old Course", 'green')
plt.xlabel("Episode")
plt.ylabel("Buoys Passed")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reinforce_vs_actorcritic_oldcourse.png")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plot_smooth([d["episode"] for d in actorcritic_data], [d.get("total_reward", 0) for d in actorcritic_data], [d.get("total_reward_std", 0) for d in actorcritic_data], "Actor-Critic", 'green')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Ticks by Episode")
plot_smooth([d["episode"] for d in actorcritic_data], [d.get("ticks", 0) for d in actorcritic_data], [d.get("ticks_std", 0) for d in actorcritic_data], "Actor-Critic", 'green')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actorcritic_comparison.png")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plot_smooth([d["episode"] for d in actorcritic_batched_data], [d.get("total_reward", 0) for d in actorcritic_batched_data], [d.get("total_reward_std", 0) for d in actorcritic_batched_data], "Actor-Critic Batched", 'blue')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Ticks by Episode")
plot_smooth([d["episode"] for d in actorcritic_batched_data], [d.get("ticks", 0) for d in actorcritic_batched_data], [d.get("ticks_std", 0) for d in actorcritic_batched_data], "Actor-Critic Batched", 'blue')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actorcritic_batched_comparison.png")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plot_smooth([d["episode"] for d in actorcritic_randomized_data], [d.get("total_reward", 0) for d in actorcritic_randomized_data], [d.get("total_reward_std", 0) for d in actorcritic_randomized_data], "Actor-Critic Randomized", 'green')
plot_smooth([d["episode"] for d in actorcritic_randomized_data], [d.get("std_total_reward", 0) for d in actorcritic_randomized_data], [d.get("std_total_reward_std", 0) for d in actorcritic_randomized_data], "Standardized Total Reward", 'orange')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Ticks by Episode")
plot_smooth([d["episode"] for d in actorcritic_randomized_data], [d.get("ticks", 0) for d in actorcritic_randomized_data], [d.get("ticks_std", 0) for d in actorcritic_randomized_data], "Actor-Critic Randomized", 'green')
plot_smooth([d["episode"] for d in actorcritic_randomized_data], [d.get("std_ticks", 0) for d in actorcritic_randomized_data], [d.get("std_ticks_std", 0) for d in actorcritic_randomized_data], "Standardized Ticks", 'orange')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actorcritic_randomized_comparison.png")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plot_smooth([d["episode"] for d in actorcritic_2buoy], [d.get("total_reward", 0) for d in actorcritic_2buoy], [d.get("total_reward_std", 0) for d in actorcritic_2buoy], "Actor-Critic 2 Buoy", 'green')
plot_smooth([d["episode"] for d in actorcritic_2buoy], [d.get("std_total_reward", 0) for d in actorcritic_2buoy], [d.get("std_total_reward_std", 0) for d in actorcritic_2buoy], "Standardized Total Reward", 'orange')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)

# plt.subplot(1, 2, 2)
# plt.title("Ticks by Episode")
# plot_smooth([d["episode"] for d in actorcritic_2buoy], [d.get("ticks", 0) for d in actorcritic_2buoy], [d.get("ticks_std", 0) for d in actorcritic_2buoy], "Actor-Critic 2 Buoy", 'green')
# plot_smooth([d["episode"] for d in actorcritic_2buoy], [d.get("std_ticks", 0) for d in actorcritic_2buoy], [d.get("std_ticks_std", 0) for d in actorcritic_2buoy], "Standardized Ticks", 'orange')
# plt.xlabel("Episode")
# plt.ylabel("Ticks")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("actorcritic_2buoy_comparison.png")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Standardized Total Reward by Episode")
plot_smooth([d["episode"] for d in actorcritic_data], [d.get("total_reward", 0) for d in actorcritic_data], [d.get("total_reward_std", 0) for d in actorcritic_data], "Actor-Critic", 'green')
plot_smooth([d["episode"] for d in actorcritic_batched_data], [d.get("total_reward", 0) for d in actorcritic_batched_data], [d.get("total_reward_std", 0) for d in actorcritic_batched_data], "Actor-Critic Batched", 'blue')
plot_smooth([d["episode"] for d in actorcritic_randomized_data], [d.get("std_total_reward", 0) for d in actorcritic_randomized_data], [d.get("std_total_reward_std", 0) for d in actorcritic_randomized_data], "Actor-Critic Randomized", 'orange')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Standardized Ticks by Episode")
plot_smooth([d["episode"] for d in actorcritic_data], [d.get("ticks", 0) for d in actorcritic_data], [d.get("ticks_std", 0) for d in actorcritic_data], "Actor-Critic", 'green')
plot_smooth([d["episode"] for d in actorcritic_batched_data], [d.get("ticks", 0) for d in actorcritic_batched_data], [d.get("ticks_std", 0) for d in actorcritic_batched_data], "Actor-Critic Batched", 'blue')
plot_smooth([d["episode"] for d in actorcritic_randomized_data], [d.get("std_ticks", 0) for d in actorcritic_randomized_data], [d.get("std_ticks_std", 0) for d in actorcritic_randomized_data], "Actor-Critic Randomized", 'orange')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actorcritic_comparison_all.png")