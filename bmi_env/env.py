import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class BMIMgmtEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 csv_path: str = "dataset.csv",   # default
                 max_days: int = 56,
                 target_bmi_low: float = 22.0,
                 target_bmi_high: float = 27.0,
                 seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # Resolve path relative to project root
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)

        # Load dataset
        self.data = pd.read_csv(csv_path)

        # --- Light cleaning to prevent NaNs/infs later ---
        # Ensure required columns exist
        required_cols = [
            "age","gender","bmi","daily_steps","hours_sleep","stress_level",
            "activity_type","blood_pressure_systolic","blood_pressure_diastolic",
            "smoking_status","fitness_level","intensity","health_condition"
        ]
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Encode categorical values
        self.data["gender"] = self.data["gender"].map({"M": 1, "F": 0})
        self.data["intensity"] = self.data["intensity"].map({"Low": 0, "Medium": 1, "High": 2})
        self.data["smoking_status"] = self.data["smoking_status"].map({"Never": 0, "Former": 1, "Current": 2})
        self.data["health_condition"] = self.data["health_condition"].map({"None": 0, "Hypertension": 1, "Diabetes": 2})

        # Encode activity type (map for simplicity)
        self.data["activity_type"] = self.data["activity_type"].fillna("Unknown")
        unique_activities = {name: i for i, name in enumerate(self.data["activity_type"].unique())}
        self.data["activity_type"] = self.data["activity_type"].map(unique_activities)

        # Coerce numeric columns and fill remaining NaNs with sensible medians
        num_cols = [
            "age","bmi","daily_steps","hours_sleep","stress_level",
            "activity_type","blood_pressure_systolic","blood_pressure_diastolic",
            "smoking_status","fitness_level","gender"
        ]
        for c in num_cols:
            self.data[c] = pd.to_numeric(self.data[c], errors="coerce")
        self.data[num_cols] = self.data[num_cols].fillna(self.data[num_cols].median(numeric_only=True))

        # Observation space:
        # [age, gender, bmi, daily_steps, hours_sleep, stress_level,
        #  activity_type, systolic_bp, diastolic_bp, smoking_status, fitness_level]
        n_act_vals = max(1, len(unique_activities))
        low = np.array([18, 0, 15.0, 0, 0, 0, 0, 80, 40, 0, 0], dtype=np.float32)
        high = np.array([80, 1, 45.0, 30000, 12, 10, n_act_vals - 1, 200, 120, 2, 5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions
        self.action_space = spaces.Discrete(7)

        self.max_days = max_days
        self.target_bmi_low = target_bmi_low
        self.target_bmi_high = target_bmi_high

        self.state = None
        self.day = 0

    def seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def _sample_patient(self):
        row = self.data.sample(1, random_state=int(self.rng.integers(1_000_000))).iloc[0]
        obs = np.array([
            row["age"],
            row["gender"],
            row["bmi"],
            row["daily_steps"],
            row["hours_sleep"],
            row["stress_level"],
            row["activity_type"],
            row["blood_pressure_systolic"],
            row["blood_pressure_diastolic"],
            row["smoking_status"],
            row["fitness_level"]
        ], dtype=np.float32)

        # Clip to space & replace non-finite
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)
        self.state = self._sample_patient()
        self.day = 0
        info = {}
        return self.state.copy(), info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action."

        # unpack state
        (age, gender, bmi, steps, sleep, stress,
         activity, sys_bp, dia_bp, smoking, fitness) = self.state

        # Apply simple action effects
        if action == 0:   # walking
            steps += 1000
            bmi -= 0.02
            fitness += 0.05
        elif action == 1: # stretching
            stress = max(0, float(stress) - 0.5)
        elif action == 2: # structured exercise
            steps += 2000
            bmi -= 0.05
            fitness += 0.1
        elif action == 3: # portion control
            bmi -= 0.03
        elif action == 4: # high-volume foods
            bmi -= 0.01
        elif action == 5: # track calories
            stress = max(0, float(stress) - 0.2)
            bmi -= 0.02
        elif action == 6: # sleep
            sleep = float(np.clip(float(sleep) + self.rng.normal(0.5, 0.2), 5.0, 10.0))

        # Simulate small blood pressure drift
        sys_bp = float(np.clip(float(sys_bp) + self.rng.normal(0, 1), 90, 200))
        dia_bp = float(np.clip(float(dia_bp) + self.rng.normal(0, 1), 60, 120))

        # Natural BMI drift
        bmi = float(np.clip(float(bmi) + self.rng.normal(0.0, 0.01), 15.0, 45.0))

        # Fitness capped
        fitness = float(np.clip(float(fitness), 0.0, 5.0))

        # New state
        self.state = np.array([
            age, gender, bmi, steps, sleep, stress,
            activity, sys_bp, dia_bp, smoking, fitness
        ], dtype=np.float32)

        # Clip to observation bounds and sanitize
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        self.state = np.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.day += 1

        # ------------------------
        # Reward formula (UPDATED + clipped)
        # ------------------------
        epsilon = 1e-6

        # BMI component (target ~24)
        bmi_score = -abs(float(bmi) - 24.0) / (24.0 + epsilon)

        # Activity component (scaled by activity index)
        denom_act = max(1.0, float(self.observation_space.high[6]))
        activity_score = float(activity) / denom_act

        # Blood pressure component (ideal ~120/80)
        bp_sys_score = -((float(sys_bp) - 120.0) ** 2) / 1000.0
        bp_dia_score = -((float(dia_bp) - 80.0) ** 2) / 500.0

        # Smoking component
        # 0=Never, 1=Former, 2=Current
        smoking_score = -1.0 if int(smoking) == 2 else (-0.5 if int(smoking) == 1 else 0.5)

        # Fitness component
        fitness_score = float(fitness) / 5.0

        # Steps (scaled)
        steps_score = float(steps) / 30000.0

        # Stress (penalty)
        stress_score = -0.1 * float(stress)

        # Sleep bonus
        sleep_score = 0.5 if 7.0 <= float(sleep) <= 9.0 else 0.0

        # Final weighted reward
        reward = (
            0.4 * bmi_score +
            0.15 * activity_score +
            0.15 * (bp_sys_score + bp_dia_score) +
            0.1 * fitness_score +
            0.1 * steps_score +
            0.05 * smoking_score +
            0.05 * stress_score +
            sleep_score
        )

        # Clip reward to stabilize learning (prevents huge returns → NaNs)
        reward = float(np.clip(reward, -5.0, 5.0))
        if not np.isfinite(reward):
            reward = 0.0  # last resort safety

        terminated = bool((bmi < 18.0) or (bmi > 42.0))
        truncated = bool(self.day >= self.max_days)

        info = {}
        if terminated or truncated:
            info["is_success"] = float(self.target_bmi_low <= float(bmi) <= self.target_bmi_high)

        # Final safety assertions (helpful during debugging; cheap)
        # Comment out if you prefer silent mode.
        # assert np.all(np.isfinite(self.state)), f"Invalid obs: {self.state}"
        # assert np.isfinite(reward), f"Invalid reward: {reward}"

        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        print(f"Day {self.day} | State: {self.state}")
