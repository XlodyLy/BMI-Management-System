import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from bmi_env.env import BMIMgmtEnv

np.set_printoptions(precision=2, suppress=True)

# Define labels for each state dimension (based on dataset columns)
STATE_LABELS = [
    "Age", "Gender", "BMI", "Calories", "Sleep Hours",
    "Alcohol", "Smoking", "Blood Pressure", "Fitness Level",
    "Family History", "Activity Type"
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

    for ep in range(episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0.0
        step_count = 0

        print(f"\n=== Episode {ep+1} ===")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Print state per step in a readable format
            print(f"Day {step_count} | State: {pretty_state(obs)}")

            if render:
                env.render()

        all_rewards.append(total_reward)
        final_states.append(obs)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved PPO model (without .zip extension)")
    parser.add_argument("--csv", type=str, default="dataset.csv", help="Path to dataset CSV file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render environment states")

    args = parser.parse_args()

    evaluate(model_path=args.model, csv_path=args.csv, episodes=args.episodes, render=args.render)
