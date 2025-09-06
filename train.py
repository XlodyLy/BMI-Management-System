import os
import argparse
import torch
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

from bmi_env.env import BMIMgmtEnv

# -------------------------------
# Environment Creation
# -------------------------------
def make_env(csv_path: str, max_days: int, seed: int = 42, reward_mode: str = "additive"):
    def _thunk():
        env = BMIMgmtEnv(csv_path=csv_path, max_days=max_days, seed=seed, reward_mode=reward_mode)
        return env
    return _thunk

def create_vec_env(args):
    env_fns = [make_env(args.csv, args.max_days, args.seed + i, args.reward_mode) for i in range(args.n_envs)]
    if args.use_subproc and args.n_envs > 1:
        base_env = SubprocVecEnv(env_fns)
    else:
        base_env = DummyVecEnv(env_fns)
    
    env = VecNormalize(
        base_env,
        norm_obs=args.norm_obs,
        norm_reward=args.norm_reward,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
        gamma=args.gamma
    )
    return env

# -------------------------------
# Model Creation
# -------------------------------
def create_model(env, args, device):
    algo_map = {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
        "ddpg": DDPG
    }
    AlgoClass = algo_map[args.algo.lower()]

    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load} ...")
        model = AlgoClass.load(args.load, env=env, device=device)
    else:
        print(f"Creating new {args.algo.upper()} model ...")
        model = AlgoClass(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            device=device
        )

    return model

# -------------------------------
# Training
# -------------------------------
def train_model(model, env, args):
    callbacks = []
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=args.checkpoint_dir,
            name_prefix=f"{args.algo}_bmi"
        )
        callbacks.append(checkpoint_callback)
    
    print("Training ...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    # Save final model
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        model.save(args.save)
        print(f"Model saved at {args.save}.zip")
        # Save VecNormalize stats
        vecnorm_path = args.save + "_vecnormalize.pkl"
        env.save(vecnorm_path)
        print(f"VecNormalize stats saved at {vecnorm_path}")

    env.close()

# -------------------------------
# Main
# -------------------------------
def main(args):
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Dataset not found: {args.csv}")

    set_random_seed(args.seed)
    
    # Device auto-detect
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    env = create_vec_env(args)
    model = create_model(env, args, device)
    train_model(model, env, args)

# -------------------------------
# Argument Parser
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--save", type=str, default="models/ppo_bmi_model", help="Path to save model (without .zip)")
    parser.add_argument("--load", type=str, default=None, help="Path to load existing model (.zip)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--max_days", type=int, default=56, help="Episode horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # -------------------------------
    # Vectorized environment args
    # -------------------------------
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--use_subproc", type=bool, default=False, help="Use SubprocVecEnv if True")
    parser.add_argument("--norm_obs", type=bool, default=True, help="Normalize observations")
    parser.add_argument("--norm_reward", type=bool, default=True, help="Normalize rewards")
    parser.add_argument("--clip_obs", type=float, default=10.0, help="Clip observations")
    parser.add_argument("--clip_reward", type=float, default=10.0, help="Clip rewards")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # -------------------------------
    # RL algorithm + reward mode
    # -------------------------------
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3", "ddpg"], help="RL algorithm to use")
    parser.add_argument("--reward_mode", type=str, default="additive", choices=["additive", "multiplicative"], help="Reward shaping mode")

    # -------------------------------
    # PPO hyperparameters
    # -------------------------------
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--tensorboard_log", type=str, default="./logs/")

    # -------------------------------
    # Checkpointing
    # -------------------------------
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=10000, help="Timesteps between checkpoints")

    # -------------------------------
    # Device
    # -------------------------------
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, or auto")

    args = parser.parse_args()
    main(args)
