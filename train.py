import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from bmi_env.env import BMIMgmtEnv


def make_env(csv_path: str, max_days: int, seed: int = 42):
    def _thunk():
        env = BMIMgmtEnv(csv_path=csv_path, max_days=max_days, seed=seed)
        return env
    return _thunk


def main(args):
    set_random_seed(args.seed)

    # Vectorized env + normalization (prevents NaNs from scale issues)
    base_env = DummyVecEnv([make_env(args.csv, args.max_days, args.seed)])
    env = VecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )

    # Create or load model
    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load} ...")
        model = PPO.load(args.load, env=env, device="cpu")
    else:
        print("Creating new PPO model ...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,           # lower LR = more stable
            batch_size=64,
            n_steps=512,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,               # a bit smaller entropy to avoid thrashing
            clip_range=0.2,
            clip_range_vf=None,
            n_epochs=10,
            max_grad_norm=0.3,            # stronger grad clipping
            policy_kwargs=dict(
                net_arch=[128, 128],      # modest network
                ortho_init=False          # slightly more stable in practice
            ),
            device="cpu"
        )

    # Train
    print("Training ...")
    model.learn(total_timesteps=args.timesteps)

    # Ensure save dir exists
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        # Save PPO weights
        model.save(args.save)
        print(f"Model saved at {args.save}.zip")
        # Save VecNormalize statistics (VERY important for later evaluation)
        vecnorm_path = args.save + "_vecnormalize.pkl"
        env.save(vecnorm_path)
        print(f"VecNormalize stats saved at {vecnorm_path}")

    # Clean up
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--save", type=str, default="models/ppo_bmi_model", help="Path to save model (without .zip)")
    parser.add_argument("--load", type=str, default=None, help="Path to load existing model (.zip)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--max_days", type=int, default=56, help="Episode horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)
