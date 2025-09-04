import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from bmi_env.env import BMIMgmtEnv

# Add this import (so we can quit pygame after use)
import pygame

np.set_printoptions(precision=2, suppress=True)

# Define labels for each state dimension (based on dataset columns)
STATE_LABELS = [
    "Age", "Gender", "BMI", "Daily Steps", "Sleep Hours",
    "Stress", "Activity Type", "Systolic BP", "Diastolic BP",
    "Smoking", "Fitness Level"
]

def pretty_state(state):
    """Format state values with labels for readability."""
    return {label: round(val, 2) for label, val in zip(STATE_LABELS, state)}

def evaluate(model_path: str, csv_path: str = "dataset.csv", episodes: int = 10, render: bool = False):
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}. Did you train and save it?")

    print(f"Loading model from {model_path} ...")
    model = PPO.load(model_path)

    env = BMIMgmtEnv(csv_path=csv_path)

    all_rewards = []
    success_count = 0
    final_states = []
    bmi_trajectories = []
    action_counts = np.zeros(env.action_space.n, dtype=int)

    for ep in range(episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0.0
        step_count = 0
        bmi_traj = []

        print(f"\n=== Episode {ep+1} ===")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Track BMI and actions
            bmi_traj.append(obs[2])  # BMI is index 2
            action_counts[action] += 1

            # Print state per step in a readable format
            print(f"Day {step_count} | State: {pretty_state(obs)}")

            if render:
                env.render()  # will open pygame window

        all_rewards.append(total_reward)
        final_states.append(obs)
        bmi_trajectories.append(bmi_traj)

        if "is_success" in info and info["is_success"]:
            success_count += 1

        print(f"\nEpisode {ep+1} Summary | Total Reward: {total_reward:.2f} | Success: {info.get('is_success', 0)}")

    avg_reward = np.mean(all_rewards)
    success_rate = success_count / episodes
    avg_final_state = np.mean(final_states, axis=0)

    print("\n=== Evaluation Results ===")
    print(f"Average Reward (Reward Rate): {avg_reward:.2f}")
    print(f"Success Rate: {success_rate * 100:.1f}% ({success_count}/{episodes})")
    print("Average Final State:", pretty_state(avg_final_state))

    # -----------------------
    # Visualization Section
    # -----------------------

    # Plot reward per episode
    plt.figure(figsize=(8, 4))
    plt.plot(all_rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.grid(True)
    plt.show()

    # Plot BMI trajectories
    plt.figure(figsize=(10, 5))
    for i, traj in enumerate(bmi_trajectories):
        plt.plot(traj, label=f"Ep {i+1}")
    plt.xlabel("Day")
    plt.ylabel("BMI")
    plt.title("BMI Trajectories Across Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot action distribution
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(action_counts)), action_counts)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action Distribution")
    plt.show()

    # ✅ Ensure pygame window closes after all episodes
    if render:
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved PPO model (without .zip extension)")
    parser.add_argument("--csv", type=str, default="dataset.csv", help="Path to dataset CSV file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render environment states")

    args = parser.parse_args()

    evaluate(model_path=args.model, csv_path=args.csv, episodes=args.episodes, render=args.render)
