
import os
import numpy as np
from stable_baselines3 import PPO
from bmi_env import BMIMgmtEnv

MODEL_PATH = os.path.join("models", "ppo_bmi.zip")

def run_episode(env, model, deterministic=True):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1

    success = info.get("is_success", 0.0)
    final_bmi = float(obs[2])
    return total_reward, steps, success, final_bmi

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train first by running: python train.py")
        return

    model = PPO.load(MODEL_PATH)
    env = BMIMgmtEnv(max_days=56, seed=123)

    num_episodes = 1000
    rewards, successes, finals = [], [], []

    for ep in range(num_episodes):
        ep_rew, ep_len, success, final_bmi = run_episode(env, model)
        rewards.append(ep_rew)
        successes.append(success)
        finals.append(final_bmi)
        print(f"Episode {ep+1:02d}: reward={ep_rew:.2f}, len={ep_len}, success={success:.0f}, final BMI={final_bmi:.2f}")

    success_rate = 100.0 * float(np.mean(successes)) if successes else 0.0
    print("\n=== Evaluation Summary ===")
    print(f"Episodes:       {num_episodes}")
    print(f"Avg Reward:     {np.mean(rewards):.2f}")
    print(f"Success Rate:   {success_rate:.1f}%  (BMI within target band at episode end)")
    print(f"Avg Final BMI:  {np.mean(finals):.2f}")

if __name__ == "__main__":
    main()
