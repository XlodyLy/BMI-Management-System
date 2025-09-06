import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import torch


# --- pygame import for visualization ---
try:
    import pygame
except Exception:
    pygame = None  # fallback if pygame not installed


class BMIMgmtEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 csv_path: str = "dataset.csv",   # default
                 max_days: int = 56,
                 target_bmi_low: float = 22.0,
                 target_bmi_high: float = 27.0,
                 seed: int | None = None,
                 reward_mode: str = "additive"):   # <-- NEW
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.reward_mode = reward_mode  # <-- NEW

        # Resolve path relative to project root
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)

        # Load dataset
        self.data = pd.read_csv(csv_path)

        # --- Light cleaning to prevent NaNs/infs later ---
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

        # Coerce numeric columns and fill remaining NaNs
        num_cols = [
            "age","bmi","daily_steps","hours_sleep","stress_level",
            "activity_type","blood_pressure_systolic","blood_pressure_diastolic",
            "smoking_status","fitness_level","gender"
        ]
        for c in num_cols:
            self.data[c] = pd.to_numeric(self.data[c], errors="coerce")
        self.data[num_cols] = self.data[num_cols].fillna(self.data[num_cols].median(numeric_only=True))

        # Observation space
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

        # --- pygame / render state ---
        self.window = None
        self.clock = None
        self.font = None
        self.last_action = None
        self.last_reward = 0.0
        self.action_labels = [
            "Walking", "Stretching", "Exercise",
            "Portion Ctrl", "High-Volume Foods",
            "Track Calories", "Sleep"
        ]

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

        (age, gender, bmi, steps, sleep, stress,
         activity, sys_bp, dia_bp, smoking, fitness) = self.state

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

        sys_bp = float(np.clip(float(sys_bp) + self.rng.normal(0, 1), 90, 200))
        dia_bp = float(np.clip(float(dia_bp) + self.rng.normal(0, 1), 60, 120))
        bmi = float(np.clip(float(bmi) + self.rng.normal(0.0, 0.01), 15.0, 45.0))
        fitness = float(np.clip(float(fitness), 0.0, 5.0))

        self.state = np.array([
            age, gender, bmi, steps, sleep, stress,
            activity, sys_bp, dia_bp, smoking, fitness
        ], dtype=np.float32)

        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        self.state = np.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.day += 1

        epsilon = 1e-6
        bmi_score = -abs(float(bmi) - 24.0) / (24.0 + epsilon)
        denom_act = max(1.0, float(self.observation_space.high[6]))
        activity_score = float(activity) / denom_act
        bp_sys_score = -((float(sys_bp) - 120.0) ** 2) / 1000.0
        bp_dia_score = -((float(dia_bp) - 80.0) ** 2) / 500.0
        smoking_score = -1.0 if int(smoking) == 2 else (-0.5 if int(smoking) == 1 else 0.5)
        fitness_score = float(fitness) / 5.0
        steps_score = float(steps) / 30000.0
        stress_score = -0.1 * float(stress)
        sleep_score = 0.5 if 7.0 <= float(sleep) <= 9.0 else 0.0

        # --- Reward function based on mode ---
        if self.reward_mode == "additive":
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
        elif self.reward_mode == "multiplicative":
            base = (
                0.4 * bmi_score +
                0.15 * activity_score +
                0.15 * (bp_sys_score + bp_dia_score) +
                0.1 * fitness_score +
                0.1 * steps_score
            )
            penalty = (0.05 * smoking_score) + (0.05 * stress_score) + sleep_score
            reward = base * (1.0 + penalty)
        else:
            reward = 0.0

        reward = float(np.clip(reward, -5.0, 5.0))
        if not np.isfinite(reward):
            reward = 0.0

        # Remember for rendering
        self.last_action = int(action)
        self.last_reward = float(reward)

        terminated = bool((bmi < 18.0) or (bmi > 42.0))
        truncated = bool(self.day >= self.max_days)

        info = {}
        if terminated or truncated:
            info["is_success"] = float(self.target_bmi_low <= float(bmi) <= self.target_bmi_high)

        return self.state.copy(), reward, terminated, truncated, info

    # --- Pygame rendering helpers ---
    def _init_pygame(self):
        if pygame is None:
            return
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((900, 520))
            pygame.display.set_caption("BMI Management — Agent Viewer")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 20)

    def _draw_text(self, surface, text, x, y, color=(230, 230, 230)):
        if self.font is None:
            return
        img = self.font.render(str(text), True, color)
        surface.blit(img, (x, y))

    def _draw_bar(self, surface, x, y, w, h, value, vmin, vmax, label, good_range=None):
        pygame.draw.rect(surface, (60, 60, 70), (x, y, w, h), border_radius=8)
        pct = 0.0 if vmax <= vmin else max(0.0, min(1.0, (float(value) - vmin) / (vmax - vmin)))
        color = (120, 190, 120)
        if good_range:
            low, high = good_range
            if not (low <= float(value) <= high):
                color = (200, 110, 110)
        pygame.draw.rect(surface, color, (x, y, int(w * pct), h), border_radius=8)
        self._draw_text(surface, f"{label}: {value:.2f}", x + 8, y + h + 6)

    def render(self):
        if pygame is None:
            print(f"Day {self.day} | State: {self.state} | "
                  f"Action: {self.last_action} | Reward: {self.last_reward:.3f}")
            return

        self._init_pygame()
        if self.window is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                self.window = None
                return

        self.window.fill((24, 26, 30))

        (age, gender, bmi, steps, sleep, stress,
         activity, sys_bp, dia_bp, smoking, fitness) = self.state

        header = f"Day {self.day} | Last Action: " \
                 f"{(self.action_labels[self.last_action] if self.last_action is not None else '—')} | " \
                 f"Last Reward: {self.last_reward:+.3f}"
        self._draw_text(self.window, header, 20, 16, color=(200, 220, 255))

        x0, y0, w, h, gap = 20, 60, 520, 22, 40
        y = y0

        self._draw_bar(self.window, x0, y, w, h, bmi, 15.0, 45.0, "BMI",
                       good_range=(self.target_bmi_low, self.target_bmi_high))
        y += gap
        self._draw_bar(self.window, x0, y, w, h, steps, 0, 30000, "Daily Steps"); y += gap
        self._draw_bar(self.window, x0, y, w, h, sleep, 0.0, 12.0, "Sleep Hours", good_range=(7.0, 9.0)); y += gap
        self._draw_bar(self.window, x0, y, w, h, stress, 0.0, 10.0, "Stress Level", good_range=(0.0, 3.0)); y += gap
        self._draw_bar(self.window, x0, y, w, h, fitness, 0.0, 5.0, "Fitness Level"); y += gap
        self._draw_bar(self.window, x0, y, w, h, sys_bp, 80.0, 200.0, "Systolic BP", good_range=(110.0, 130.0)); y += gap
        self._draw_bar(self.window, x0, y, w, h, dia_bp, 40.0, 120.0, "Diastolic BP", good_range=(70.0, 90.0))

        sx, sy = 580, 80
        self._draw_text(self.window, f"Age: {int(age)}", sx, sy); sy += 28
        self._draw_text(self.window, f"Gender: {'M' if int(gender)==1 else 'F'}", sx, sy); sy += 28
        self._draw_text(self.window, f"Activity Type: {int(activity)}", sx, sy); sy += 28
        self._draw_text(self.window, f"Smoking: {['Never','Former','Current'][int(smoking)]}", sx, sy); sy += 28
        sy += 10
        self._draw_text(self.window, f"Target BMI: {self.target_bmi_low:.1f}–{self.target_bmi_high:.1f}", sx, sy); sy += 28
        in_target = (self.target_bmi_low <= float(bmi) <= self.target_bmi_high)
        status = "IN RANGE ✅" if in_target else "OUT OF RANGE ❌"
        self._draw_text(self.window, f"Status: {status}", sx, sy)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if pygame is not None:
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
        self.window = None
        self.clock = None
        self.font = None
