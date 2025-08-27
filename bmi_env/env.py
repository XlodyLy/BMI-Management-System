
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BMIMgmtEnv(gym.Env):
    """
    A custom environment simulating lifestyle management for high-risk cardiovascular patients.
    Observation: [Age, Gender, BMI, Exercise(E), Diet(D), Sleep(S), MissedMeals(M), JunkFood(F), MissedExercise(P)]
    Discrete actions (7): exercise & lifestyle suggestions.
    Reward per step (day): R = α(-ΔBMI) + βE + γD + δ(S/9) - θM - κF - λP
    Episode is truncated at a max number of days or terminated if BMI goes out of safe bounds.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 max_days: int = 56,          # simulate 8 weeks
                 target_bmi_low: float = 22.0,
                 target_bmi_high: float = 27.0,
                 seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # Observation space: Age, Gender(0/1), BMI, E, D, S, M, F, P
        low = np.array([40, 0, 15.0, 0.0, 0.0, 0.0, 0, 0, 0], dtype=np.float32)
        high = np.array([70, 1, 45.0, 1.0, 1.0, 12.0, 5, 7, 7], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions: 0..6
        self.action_space = spaces.Discrete(7)

        # Config
        self.max_days = max_days
        self.target_bmi_low = target_bmi_low
        self.target_bmi_high = target_bmi_high

        # Reward weights (tunable)
        self.alpha = 5.0   # scales -ΔBMI
        self.beta  = 1.0   # E
        self.gamma = 0.7   # D
        self.delta = 0.3   # S/9
        self.theta = 0.2   # M
        self.kappa = 0.25  # F
        self.lmbda = 0.25  # P

        self.state = None
        self.day = 0

    def seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def _random_patient(self):
        age = self.rng.integers(40, 71)
        gender = self.rng.integers(0, 2)  # 0/1
        bmi = self.rng.uniform(26.0, 34.0)  # start slightly overweight
        E = 0.0
        D = float(self.rng.choice([0.0, 0.5]))
        S = float(self.rng.uniform(5.5, 8.5))
        M = float(self.rng.integers(0, 2))
        F = float(self.rng.integers(0, 3))
        P = float(self.rng.integers(0, 3))
        return np.array([age, gender, bmi, E, D, S, M, F, P], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)
        self.state = self._random_patient()
        self.day = 0
        info = {}
        return self.state.copy(), info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action."
        age, gender, bmi, E, D, S, M, F, P = self.state

        # Compliance probability increases with existing good habits
        compliance = np.clip(0.55 + 0.3*E + 0.15*D, 0.2, 0.95)
        did_comply = self.rng.random() < compliance

        # Natural habit decay (if not worked on)
        E = max(0.0, E - 0.04)
        D = max(0.0, D - 0.03)

        # Apply action effects (if complied)
        if action == 0:   # Light walking
            if did_comply:
                E = np.clip(E + 0.25, 0.0, 1.0)
                P = max(0.0, P - 1.0)
                bmi -= 0.05
            else:
                P = min(7.0, P + 0.5)
        elif action == 1: # Chair stretching
            if did_comply:
                E = np.clip(E + 0.15, 0.0, 1.0)
                P = max(0.0, P - 0.5)
        elif action == 2: # Structured exercise
            if did_comply:
                E = np.clip(E + 0.35, 0.0, 1.0)
                P = max(0.0, P - 1.0)
                bmi -= 0.08
            else:
                P = min(7.0, P + 1.0)
        elif action == 3: # Portion control
            if did_comply:
                D = np.clip(D + 0.35, 0.0, 1.0)
                M = max(0.0, M - 0.5)
                bmi -= 0.04
        elif action == 4: # High-volume low-calorie foods
            if did_comply:
                D = np.clip(D + 0.25, 0.0, 1.0)
                F = max(0.0, F - 1.0)
        elif action == 5: # Track/adjust calories
            if did_comply:
                D = np.clip(D + 0.2, 0.0, 1.0)
                M = max(0.0, M - 1.0)
                F = max(0.0, F - 0.5)
        elif action == 6: # Sleep 7-9 hours
            if did_comply:
                S = float(np.clip(self.rng.normal(8.0, 0.3), 7.0, 9.0))
            else:
                S = float(np.clip(S + self.rng.normal(0.0, 0.5), 5.0, 10.0))

        # Lifestyle dynamics affecting BMI (daily)
        sleep_bonus = -0.02 if 7.0 <= S <= 9.0 else 0.02
        bmi_change = (
            -0.08 * E    # exercise reduces BMI
            -0.05 * D    # good diet reduces BMI
            + sleep_bonus
            + 0.01 * F   # junk food increases BMI
            + 0.005 * P  # missed exercise increases BMI
            + self.rng.normal(0.0, 0.01)  # noise
        )
        new_bmi = float(np.clip(bmi + bmi_change, 15.0, 45.0))

        # Reward: ΔBMI = new - old → negative is good (weighting by alpha)
        delta_bmi = new_bmi - float(bmi)
        reward = (
            self.alpha * (-delta_bmi)
            + self.beta * E
            + self.gamma * D
            + self.delta * (S / 9.0)
            - self.theta * M
            - self.kappa * F
            - self.lmbda * P
        )

        # Update state/day
        self.state = np.array([age, gender, new_bmi, E, D, S, M, F, P], dtype=np.float32)
        self.day += 1

        # Terminate for unsafe BMI; truncate at horizon
        unsafe = bool(new_bmi < 18.0 or new_bmi > 42.0)
        terminated = unsafe
        truncated = self.day >= self.max_days

        info = {}
        if terminated or truncated:
            # success: BMI ended within target band
            info["is_success"] = float(self.target_bmi_low <= new_bmi <= self.target_bmi_high)

        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self):
        pass
