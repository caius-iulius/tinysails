import matplotlib.pyplot as plt
import csv

reinforce_small_data = []
with open('../logs/reinforce_small.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        reinforce_small_data.append({
            "episode": int(row["episode"]),
            "total_reward": float(row["total_reward"]),
            "ticks": int(row["ticks"]),
            "buoys_passed": int(row["buoys_passed"])
        })

reinforce_big_data = []
with open('../logs/reinforce_big.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        reinforce_big_data.append({
            "episode": int(row["episode"]),
            "total_reward": float(row["total_reward"]),
            "ticks": int(row["ticks"]),
            "buoys_passed": int(row["buoys_passed"])
        })

actorcritic_oldcourse_data = []
with open('../logs/actorcritic_logs_oldcourse.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        actorcritic_oldcourse_data.append({
            "episode": int(row["episode"]),
            "total_reward": float(row["total_reward"]),
            "ticks": int(row["ticks"]),
            "buoys_passed": int(row["buoys_passed"]),
            "learning_rate": float(row["learning_rate"]),
            "entropy_coef": float(row["entropy_coef"])
        })

actorcritic_data = []
with open('../logs/actorcritic.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        actorcritic_data.append({
            "episode": int(row["episode"]),
            "total_reward": float(row["total_reward"]),
            "ticks": int(row["ticks"]),
            "buoys_passed": int(row["buoys_passed"]),
            "learning_rate": float(row["learning_rate"]),
            "entropy_coef": float(row["entropy_coef"])
        })

actorcritic_batched_data = []
with open('../logs/actorcritic_batched.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        actorcritic_batched_data.append({
            "episode": int(row["episode"]),
            "total_reward": float(row["total_reward"]),
            "ticks": int(row["ticks"]),
            "buoys_passed": int(row["buoys_passed"]),
            "learning_rate": float(row["learning_rate"]),
            "entropy_coef": float(row["entropy_coef"])
        })

actorcritic_randomized_data = []
with open('../logs/actorcritic_randomized.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        actorcritic_randomized_data.append({
            "episode": int(row["episode"]),
            "total_reward": float(row["total_reward"]),
            "ticks": int(row["ticks"]),
            "buoys_passed": int(row["buoys_passed"]),
            "std_total_reward": float(row["std_total_reward"]),
            "std_ticks": float(row["std_ticks"]),
            "std_buoys_passed": float(row["std_buoys_passed"]),
            "learning_rate": float(row["learning_rate"]),
            "entropy_coef": float(row["entropy_coef"])
        })

# generates graph of reward and time by episode of big and small reinforce
# shows nothign if not available for that episode

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plt.plot([d["episode"] for d in reinforce_big_data], [d["total_reward"] for d in reinforce_big_data], label="Reinforce Big", color='orange')
plt.plot([d["episode"] for d in reinforce_small_data], [d["total_reward"] for d in reinforce_small_data], label="Reinforce Small", color='blue')

#save plot
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Ticks by Episode")
plt.plot([d["episode"] for d in reinforce_big_data], [d["ticks"] for d in reinforce_big_data], label="Reinforce Big", color='orange')
plt.plot([d["episode"] for d in reinforce_small_data], [d["ticks"] for d in reinforce_small_data], label="Reinforce Small", color='blue')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.tight_layout()
plt.savefig("reinforce_comparison.png")

# comparison of reinforce_small and old course actor critic (ticks)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Ticks by Episode")
plt.plot([d["episode"] for d in reinforce_small_data], [d["ticks"] for d in reinforce_small_data], label="Reinforce Small", color='blue')
plt.plot([d["episode"] for d in actorcritic_oldcourse_data], [d["ticks"] for d in actorcritic_oldcourse_data], label="Actor-Critic Old Course", color='green')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Buoys Passed by Episode")
plt.plot([d["episode"] for d in reinforce_small_data], [d["buoys_passed"] for d in reinforce_small_data], label="Reinforce Small", color='blue')
plt.plot([d["episode"] for d in actorcritic_oldcourse_data], [d["buoys_passed"] for d in actorcritic_oldcourse_data], label="Actor-Critic Old Course", color='green')
plt.xlabel("Episode")
plt.ylabel("Buoys Passed")
plt.legend()
plt.tight_layout()
plt.savefig("reinforce_vs_actorcritic_oldcourse.png")

# graph of rewards, ticks, and buoys passed by episode for actorcritic
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plt.plot([d["episode"] for d in actorcritic_data], [d["total_reward"] for d in actorcritic_data], label="Actor-Critic", color='green')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Ticks by Episode")
plt.plot([d["episode"] for d in actorcritic_data], [d["ticks"] for d in actorcritic_data], label="Actor-Critic", color='green')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.tight_layout()
plt.savefig("actorcritic_comparison.png")

# graph of rewards, ticks, and buoys passed by episode for actorcritic-batched
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plt.plot([d["episode"] for d in actorcritic_batched_data], [d["total_reward"] for d in actorcritic_batched_data], label="Actor-Critic", color='green')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Ticks by Episode")
plt.plot([d["episode"] for d in actorcritic_batched_data], [d["ticks"] for d in actorcritic_batched_data], label="Actor-Critic", color='green')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.tight_layout()
plt.savefig("actorcritic_batched_comparison.png")

# graph of rewards, standard rewards, standard ticks for actorcritic-randomized
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Total Reward by Episode")
plt.plot([d["episode"] for d in actorcritic_randomized_data], [d["total_reward"] for d in actorcritic_randomized_data], label="Actor-Critic Randomized", color='green')
plt.plot([d["episode"] for d in actorcritic_randomized_data], [d["std_total_reward"] for d in actorcritic_randomized_data], label="Standardized Total Reward", color='orange')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Ticks by Episode")
plt.plot([d["episode"] for d in actorcritic_randomized_data], [d["ticks"] for d in actorcritic_randomized_data], label="Actor-Critic Randomized", color='green')
plt.plot([d["episode"] for d in actorcritic_randomized_data], [d["std_ticks"] for d in actorcritic_randomized_data], label="Standardized Ticks", color='orange')
plt.xlabel("Episode")
plt.ylabel("Ticks")
plt.legend()
plt.tight_layout()
plt.savefig("actorcritic_randomized_comparison.png")