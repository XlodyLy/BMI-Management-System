import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from bmi_env.env import BMIMgmtEnv

import pygame

np.set_printoptions(precision=2, suppress=True)

STATE_LABELS = [
    "Age", "Gender", "BMI", "Daily Steps", "Sleep Hours",
    "Stress", "Activity Type", "Systolic BP", "Diastolic BP",
    "Smoking", "Fitness Level"
]

def pretty_state(state):
    return {label: round(float(val), 2) for label, val in zip(STATE_LABELS, state)}

def make_env(csv_path: str, max_days: int = 56, seed: int | None = None):
    def _thunk():
        return BMIMgmtEnv(csv_path=csv_path, max_days=max_days, seed=seed)
    return _thunk

def evaluate(model_path: str, csv_path: str = "dataset.csv", episodes: int = 10, render: bool = False, max_days: int = 56):
    # check files
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}.zip")
    vecnorm_path = model_path + "_vecnormalize.pkl"
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(
            f"VecNormalize stats not found at {vecnorm_path}. "
            f"Did you run train.py (it saves this automatically)?"
        )

    print(f"Loading model from {model_path} ...")
    model = PPO.load(model_path)

    # Build vectorized env and load normalization stats
    venv = DummyVecEnv([make_env(csv_path=csv_path, max_days=max_days)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    # Handy: underlying raw env (for render + readable state)
    raw_env = venv.venv.envs[0]

    all_rewards = []
    success_count = 0
    final_states = []
    bmi_trajectories = []
    action_counts = np.zeros(raw_env.action_space.n, dtype=int)

    for ep in range(episodes):
        vobs = venv.reset()            # normalized observation (shape: [1, obs_dim])
        done = False
        total_reward = 0.0
        step_count = 0
        bmi_traj = []

        print(f"\n=== Episode {ep+1} ===")
        while not done:
            action, _ = model.predict(vobs, deterministic=True)
            vobs, rewards, dones, infos = venv.step(action)

            total_reward += float(rewards[0])
            step_count += 1
            action_counts[int(action[0])] += 1

            # Read the REAL state directly from your underlying env for logging/plots
            state = raw_env.state
            bmi_traj.append(float(state[2]))  # BMI index = 2

            print(f"Day {step_count} | State: {pretty_state(state)}")

            if render:
                raw_env.render()

            done = bool(dones[0])

        all_rewards.append(total_reward)
        final_states.append(raw_env.state.copy())
        bmi_trajectories.append(bmi_traj)

        is_success = 0.0
        if infos and isinstance(infos, (list, tuple)) and len(infos) > 0:
            is_success = float(infos[0].get("is_success", 0.0))
        print(f"\nEpisode {ep+1} Summary | Total Reward: {total_reward:.2f} | Success: {is_success}")
        success_count += int(is_success > 0.0)

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    success_rate = success_count / max(1, episodes)
    avg_final_state = np.mean(np.array(final_states), axis=0) if final_states else np.zeros(len(STATE_LABELS))

    print("\n=== Evaluation Results ===")
    print(f"Average Reward (Reward Rate): {avg_reward:.2f}")
    print(f"Success Rate: {success_rate * 100:.1f}% ({success_count}/{episodes})")
    print("Average Final State:", pretty_state(avg_final_state))

    # ---- Visuals ----
    plt.figure(figsize=(8, 4))
    plt.plot(all_rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for i, traj in enumerate(bmi_trajectories):
        plt.plot(traj, label=f"Ep {i+1}")
    plt.xlabel("Day")
    plt.ylabel("BMI")
    plt.title("BMI Trajectories Across Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(action_counts)), action_counts)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action Distribution")
    plt.show()

    if render:
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved PPO model (without .zip extension)")
    parser.add_argument("--csv", type=str, default="dataset.csv", help="Path to dataset CSV file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render environment states")
    parser.add_argument("--max_days", type=int, default=56, help="Episode horizon (match training)")
    args = parser.parse_args()

    evaluate(model_path=args.model, csv_path=args.csv, episodes=args.episodes, render=args.render, max_days=args.max_days)
