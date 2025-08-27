import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from bmi_env import BMIMgmtEnv   # make sure bmi_env.py is in the same folder
from utils.callbacks import EpisodicCSVLogger

# Directories for logs and models
LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(seed=42):
    def _init():
        env = BMIMgmtEnv(max_days=56, seed=seed)  # 
        env = Monitor(env)  # Records episode returns/lengths
        return env
    return _init

def main():
    # Wrap environment for compatibility with PPO
    env = DummyVecEnv([make_env(seed=42)])

    # Define PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        seed=42,
    )

    # Log training results to CSV
    csv_logger = EpisodicCSVLogger(csv_path=os.path.join(LOG_DIR, "episode_metrics.csv"))

    # Train the model
    total_timesteps = 500000   # 🔹 adjust if needed (more timesteps = better learning)
    model.learn(total_timesteps=total_timesteps, callback=csv_logger)

    # Save model
    save_path = os.path.join(MODEL_DIR, "ppo_bmi")
    model.save(save_path)
    print(f"✅ Saved model to {save_path}.zip")

if __name__ == "__main__":
    main()
