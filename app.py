# app.py
import streamlit as st
import sqlite3
import os
import hashlib
import datetime
import numpy as np
import pandas as pd
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from bmi_env.env import BMIMgmtEnv

# ---------- Config ----------
DB_PATH = "user_data.db"
MODEL_BASE = "models/ppo_bmi_model"  # no .zip extension
MODEL_ZIP = MODEL_BASE + ".zip"
VEC_PATH = MODEL_BASE + "_vecnormalize.pkl"

# ---------- Utilities: DB & Password Hashing ----------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        full_name TEXT,
        salt TEXT,
        pw_hash TEXT,
        age INTEGER,
        gender TEXT,
        baseline_bmi REAL,
        fitness INTEGER,
        smoking TEXT,
        activity_level TEXT,
        usual_sleep REAL,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        date TEXT,
        bmi REAL,
        steps INTEGER,
        sleep REAL,
        stress INTEGER,
        activity_type INTEGER,
        sys_bp INTEGER,
        dia_bp INTEGER,
        smoking INTEGER,
        fitness INTEGER,
        heart_rate INTEGER,
        action INTEGER,
        recommendation TEXT,
        created_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str):
    salt = os.urandom(16)
    hash_bytes = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 150000)
    return salt.hex(), hash_bytes.hex()

def verify_password(password: str, salt_hex: str, hash_hex: str):
    salt = bytes.fromhex(salt_hex)
    check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 150000).hex()
    return check == hash_hex

# ---------- DB actions ----------
def create_user(username, full_name, password, age, gender, baseline_bmi, fitness):
    conn = get_conn()
    cur = conn.cursor()
    salt, pw_hash = hash_password(password)
    cur.execute("""
    INSERT INTO users (username, full_name, salt, pw_hash, age, gender, baseline_bmi, fitness, smoking, activity_level, usual_sleep, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, full_name, salt, pw_hash, age, gender, baseline_bmi, fitness, None, None, None, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def update_profile(user_id, smoking, activity_level, usual_sleep):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    UPDATE users
    SET smoking = ?, activity_level = ?, usual_sleep = ?
    WHERE id = ?
    """, (smoking, activity_level, usual_sleep, user_id))
    conn.commit()
    conn.close()

def get_user_by_username(username):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def insert_log(user_id, payload: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO logs (user_id, date, bmi, steps, sleep, stress, activity_type, sys_bp, dia_bp,
                      smoking, fitness, heart_rate, action, recommendation, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        payload.get("date"),
        payload.get("bmi"),
        payload.get("steps"),
        payload.get("sleep"),
        payload.get("stress"),
        payload.get("activity_type"),
        payload.get("sys_bp"),
        payload.get("dia_bp"),
        payload.get("smoking"),
        payload.get("fitness"),
        payload.get("heart_rate"),
        payload.get("action"),
        payload.get("recommendation"),
        datetime.datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def get_logs_for_user(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM logs WHERE user_id = ? ORDER BY date ASC", (user_id,))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    return df

# ---------- Load RL Model ----------
def safe_load_model():
    if not os.path.exists(MODEL_ZIP) or not os.path.exists(VEC_PATH):
        return None, None
    try:
        model = PPO.load(MODEL_BASE)
        venv = DummyVecEnv([lambda: BMIMgmtEnv(csv_path="dataset.csv", max_days=56)])
        venv = VecNormalize.load(VEC_PATH, venv)
        venv.training = False
        venv.norm_reward = False
        return model, venv
    except Exception as e:
        st.warning(f"Model load failed: {e}")
        return None, None

def normalize_obs(obs: np.ndarray, venv: Optional[VecNormalize]):
    if venv is None:
        return obs
    mean = venv.obs_rms.mean
    var = venv.obs_rms.var
    eps = getattr(venv, "epsilon", 1e-8)
    obs_norm = (obs - mean) / np.sqrt(var + eps)
    clip_val = getattr(venv, "clip_obs", None)
    if clip_val is not None:
        obs_norm = np.clip(obs_norm, -clip_val, clip_val)
    return obs_norm

# ---------- Plan Generator ----------
def generate_full_plan(action, state_dict):
    action_texts = {
        0: "Encourage light walking (15–30 mins)",
        1: "Do chair-based stretching & relaxation",
        2: "Try structured low-intensity exercises",
        3: "Practice portion control in meals",
        4: "Eat high-volume, low-calorie foods",
        5: "Track and adjust calorie intake",
        6: "Ensure 7–9 hours of quality sleep"
    }
    main_focus = action_texts.get(action, "Keep a balanced routine")

    plan = {"main_focus": main_focus, "morning": [], "afternoon": [], "evening": [], "notes": []}

    bmi = state_dict["bmi"]
    steps = state_dict["steps"]
    sleep = state_dict["sleep"]
    stress = state_dict["stress"]
    fitness = state_dict["fitness"]
    smoking = state_dict["smoking"]
    sys_bp = state_dict["sys_bp"]
    dia_bp = state_dict["dia_bp"]
    heart_rate = state_dict["heart_rate"]

    if bmi >= 30:
        plan["morning"].append("Start with a fiber-rich breakfast (e.g., oats + fruit).")
    else:
        plan["morning"].append("Protein-rich breakfast (eggs / greek yoghurt + whole grains).")
    if steps < 5000:
        plan["morning"].append("Short brisk walk: 15–20 minutes.")

    if stress >= 6:
        plan["afternoon"].append("Take 10 minutes for breathing exercises or meditation.")
    plan["afternoon"].append("Balanced lunch: vegetables + lean protein.")
    if sys_bp > 130:
        plan["afternoon"].append("Limit salty snacks during lunch.")

    if fitness <= 2:
        plan["evening"].append("Light bodyweight circuit.")
    else:
        plan["evening"].append("Moderate 30-min cardio.")
    if sleep < 7:
        plan["evening"].append("No screens 1h before bed.")

    if smoking == 2:
        plan["notes"].append("Consider reducing smoking gradually.")
    if heart_rate > 95:
        plan["notes"].append("High resting heart rate — monitor closely.")
    if dia_bp > 90:
        plan["notes"].append("Elevated diastolic pressure — consult physician.")
    

    return plan

# ---------- Streamlit App ----------
st.set_page_config(page_title="BMI Management Platform ", layout="wide")
st.title("🏥 Intelligent BMI Management Platform For Heart Patients")
st.write("Register / login, log your day, and get a personalized plan")

init_db()
model, venv = safe_load_model()

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---------- Auth ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "profile_setup_user" not in st.session_state:
    st.session_state.profile_setup_user = None

if st.session_state.user is None and st.session_state.profile_setup_user is None:
    col1, col2 = st.columns(2)

    with col1:
        st.header("🔐 Login")
        uname = st.text_input("Username (login)", key="login_username")
        pwd = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user = get_user_by_username(uname)
            if user and verify_password(pwd, user["salt"], user["pw_hash"]):
                st.success("Logged in")
                st.session_state.user = user
                safe_rerun()
            else:
                st.error("Invalid credentials")

    with col2:
        st.header("📝 Register")
        new_user = st.text_input("Choose a username", key="reg_username")
        full_name = st.text_input("Full name", key="reg_fullname")
        new_pw = st.text_input("Password", type="password", key="reg_password")
        age_in = st.slider("Age", 18, 80, 30, key="reg_age")
        gender_in = st.selectbox("Gender", ["Male", "Female"], key="reg_gender")
        base_bmi = st.number_input("Baseline BMI", 15.0, 45.0, 25.0, key="reg_bmi")
        base_fit = st.slider("Fitness (0-5)", 0, 5, 2, key="reg_fit")
        if st.button("Create account"):
            if get_user_by_username(new_user):
                st.error("Username already exists")
            else:
                create_user(new_user, full_name or new_user, new_pw, age_in, gender_in, base_bmi, base_fit)
                new_user_row = get_user_by_username(new_user)
                st.session_state.profile_setup_user = new_user_row
                st.success("Account created — please complete your profile")
                safe_rerun()

# ---------- Profile Setup ----------
elif st.session_state.profile_setup_user is not None:
    u = st.session_state.profile_setup_user
    st.header("⚙️ Complete Your Profile")
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    activity = st.radio("Activity Level", ["Low", "Moderate", "High"])
    usual_sleep = st.slider("Usual Sleep Hours", 4, 12, 7)

    if st.button("Save Profile"):
        update_profile(u["id"], smoking, activity, usual_sleep)
        st.session_state.user = get_user_by_id(u["id"])
        st.session_state.profile_setup_user = None
        st.success("Profile completed ✅")
        safe_rerun()

# ---------- Main app after login ----------
elif st.session_state.user:
    user = st.session_state.user

    # Check if profile is incomplete
    if not user.get("smoking") or not user.get("activity_level") or not user.get("usual_sleep"):
        st.header("⚙️ Complete Your Profile")
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        activity = st.radio("Activity Level", ["Low", "Moderate", "High"])
        usual_sleep = st.slider("Usual Sleep Hours", 4, 12, 7)

        if st.button("Save Profile"):
            update_profile(user["id"], smoking, activity, usual_sleep)
            st.session_state.user = get_user_by_id(user["id"])
            st.success("Profile completed ✅")
            safe_rerun()

    else:
        # Continue to main app only if profile is complete
        st.sidebar.write(f"Signed in as **{user['full_name']}** ({user['username']})")
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            safe_rerun()

    # Daily Log
    st.header("📥 Daily Log")
    today = datetime.date.today().isoformat()
    with st.form("daily_form"):
        bmi_in = st.number_input("BMI", 10.0, 50.0, float(user.get("baseline_bmi", 25.0)))
        steps_in = st.number_input("Daily steps", 0, 30000, value=5000, step=500)
        sleep_hours = st.selectbox("Sleep hours", list(range(1, 13)), index=6)
        stress_in = st.slider("Stress (0-10)", 0, 10, 4)
        activity_in = st.selectbox("Activity type (0 sedentary, 1 light, 2 moderate, 3 active)", [0,1,2,3], index=1)
        sys_in = st.number_input("Systolic BP", 80, 220, 120)
        dia_in = st.number_input("Diastolic BP", 40, 140, 80)
        smoking_in = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        fitness_in = st.slider("Fitness level (0-5)", 0, 5, int(user.get("fitness",2)))
        hr_in = st.number_input("Resting heart rate (bpm)  **Check by placing hand on wrist or neck**", 30, 180, 75)
        submitted = st.form_submit_button("Save & Get Plan")

    if submitted:
        state = np.array([
            float(user.get("age", 30)),
            1 if user.get("gender","Male") == "Male" else 0,
            float(bmi_in), float(steps_in), (sleep_hours), float(stress_in),
            int(activity_in), float(sys_in), float(dia_in),
            int(["Never","Former","Current"].index(smoking_in)),
            float(fitness_in)
        ], dtype=np.float32).reshape(1, -1)

        if model is not None and venv is not None:
            norm = normalize_obs(state, venv)
            try:
                action_arr, _ = model.predict(norm, deterministic=True)
                action = int(np.array(action_arr).ravel()[0])
            except Exception as e:
                st.warning(f"Model inference failed: {e}")
                action = None
        else:
            action = None

        state_dict = {
            "bmi": bmi_in, "steps": steps_in, "sleep": sleep_hours, "stress": stress_in,
            "activity_type": activity_in, "sys_bp": sys_in, "dia_bp": dia_in,
            "smoking": ["Never","Former","Current"].index(smoking_in),
            "fitness": fitness_in, "heart_rate": hr_in
        }
        if action is None:
            if bmi_in > 28: action = 3
            elif steps_in < 3000: action = 0
            else: action = 2

        plan = generate_full_plan(action, state_dict)

        payload = {
            "date": today, "bmi": bmi_in, "steps": steps_in, "sleep": sleep_hours, "stress": stress_in,
            "activity_type": activity_in, "sys_bp": sys_in, "dia_bp": dia_in,
            "smoking": ["Never","Former","Current"].index(smoking_in),
            "fitness": fitness_in, "heart_rate": hr_in, "action": action,
            "recommendation": plan["main_focus"]
        }
        insert_log(user["id"], payload)
        st.success("Saved today’s log and generated your plan ✅")

        st.subheader("🏷️ Main Focus")
        st.write(plan["main_focus"])
        st.subheader("📅 Full Daily Plan")
        st.markdown("**Morning**")
        for t in plan["morning"]: st.write("- " + t)
        st.markdown("**Afternoon**")
        for t in plan["afternoon"]: st.write("- " + t)
        st.markdown("**Evening**")
        for t in plan["evening"]: st.write("- " + t)
        if plan["notes"]:
            st.subheader("⚠️ Notes")
            for n in plan["notes"]: st.write("- " + n)

    # Dashboard
    st.header("📈 Your History & Dashboard")
    logs_df = get_logs_for_user(user["id"])
    if logs_df.empty:
        st.info("No logs yet. Add your first daily log to see trends.")
    else:
        logs_df["date"] = pd.to_datetime(logs_df["date"])
        logs_df = logs_df.sort_values("date")
        st.dataframe(logs_df[['date','bmi','steps','sleep','stress','action']].tail(50))

        st.subheader("BMI over time")
        st.line_chart(logs_df.set_index("date")["bmi"])
        st.subheader("Steps over time")
        st.line_chart(logs_df.set_index("date")["steps"])
        st.subheader("Sleep & Stress")
        st.line_chart(logs_df.set_index("date")[["sleep","stress"]])