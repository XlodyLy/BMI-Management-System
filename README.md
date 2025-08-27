
# Intelligent BMI Management System (RL + PPO) — Starter Project

This repo gives you a **working custom Gymnasium environment** that simulates a patient's lifestyle context,
and a **PPO training/evaluation** pipeline using Stable-Baselines3.

---

## Phase 0 — Install Python
1) Install **Python 3.10 or 3.11** from https://www.python.org/downloads/  
   ✅ Tick "Add Python to PATH" during install (Windows).

## Phase 1 — Get the project
**Option A (ZIP)**: download `bmi_rl_project.zip` from ChatGPT and extract, e.g. to `C:\Users\<you>\ML\bmi_rl_project`  
**Option B (Manual)**: recreate this folder layout and paste file contents accordingly:

```
bmi_rl_project/
  bmi_env/
    __init__.py
    env.py
  utils/
    __init__.py
    callbacks.py
  logs/
  models/
  train.py
  evaluate.py
  requirements.txt
  README.md
```

## Phase 2 — Create & activate a virtual environment
Open **PowerShell** inside the `bmi_rl_project` folder (Shift + Right Click → “Open PowerShell window here”) and run:

**Windows (PowerShell)**
```ps1
python -m venv .venv
# If activation errors: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD)**
```bat
python -m venv .venv
.\.venv\Scriptsctivate.bat
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> You only create this **once per project**. Next sessions: just activate again.

## Phase 3 — Install dependencies
With the venv **activated**, run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
If PyTorch fails, get the correct command from https://pytorch.org/get-started/locally/. CPU-only is fine.

## Phase 4 — Train PPO
```bash
python train.py
```
- Trains PPO on your custom environment
- Saves model to `models/ppo_bmi.zip`
- Logs episodic reward/length to `logs/episode_metrics.csv`

## Phase 5 — Evaluate
```bash
python evaluate.py
```
- Loads `models/ppo_bmi.zip`
- Runs multiple episodes
- Prints **Success Rate** (BMI in target band at the end), average reward, and average final BMI

## What’s inside
- **State (observation):** `[Age, Gender, BMI, Exercise, Diet, Sleep, MissedMeals, JunkFood, MissedExercise]`
- **Actions (0–6):**
  0 Light walking; 1 Chair stretching; 2 Structured exercise;
  3 Portion control; 4 High-volume low-calorie foods; 5 Track/adjust calories; 6 Sleep 7–9 hours
- **Reward:** `R = α(-ΔBMI) + βE + γD + δ(S/9) − θM − κF − λP` (weights are tunable in `env.py`)
- **Episode end:** unsafe BMI or max simulated days

Happy training!
