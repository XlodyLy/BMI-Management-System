import streamlit as st
import numpy as np
from stable_baselines3 import PPO

# --- Load PPO Model ---
MODEL_PATH = "models/ppo_bmi_model.zip"
model = PPO.load(MODEL_PATH)

st.set_page_config(page_title="🏥 Intelligent BMI Management System", layout="centered")
st.title("🏥 Intelligent BMI Management System")
st.write("Get AI-powered personalized daily recommendations for BMI management and overall health.")

# --- User Information ---
st.header("👤 Personal Information")
name = st.text_input("Full Name")
age = st.slider("Age", 18, 80, 30)
gender = st.radio("Gender", ["Male", "Female"])

# --- Health Metrics ---
st.header("🩺 Health Metrics")
bmi = st.slider("Body Mass Index (BMI)", 15.0, 45.0, 25.0)
steps = st.number_input("Average Daily Steps", min_value=0, max_value=30000, value=5000, step=500)
sleep = st.slider("Average Sleep (hours per night)", 4.0, 12.0, 7.0)
stress = st.slider("Stress Level (0 = none, 10 = max)", 0, 10, 5)

st.markdown("---")
st.subheader("💓 Measure Your Heart Rate")
st.write("👉 Place two fingers on your wrist or neck, count beats for **15 seconds**, and multiply by 4.")
heart_rate = st.slider("Average Resting Heart Rate (bpm)", 50, 120, 75)

# --- Lifestyle Factors ---
st.header("🏃 Lifestyle & Habits")
activity_type = st.selectbox("Activity Type", [0, 1, 2, 3], help="0 = sedentary, 1 = light, 2 = moderate, 3 = active")
smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
fitness = st.slider("Fitness Level (0 = poor, 5 = excellent)", 0, 5, 2)

# --- Blood Pressure ---
st.header("🩸 Blood Pressure")
sys_bp = st.slider("Systolic BP (mmHg)", 80, 200, 120)
dia_bp = st.slider("Diastolic BP (mmHg)", 40, 120, 80)

# --- Build state vector ---
state = np.array([
    age,
    1 if gender == "Male" else 0,  # gender encoding (1=Male,0=Female)
    bmi,
    steps,
    sleep,
    stress,
    activity_type,
    sys_bp,
    dia_bp,
    ["Never","Former","Current"].index(smoking),
    fitness
], dtype=np.float32)

# --- Predict Action ---
if st.button("🔍 Generate Daily Health Plan"):
    action, _ = model.predict(state, deterministic=True)
    action = int(action)

    # RL-driven "main focus"
    actions_map = {
        0: "🏃 Encourage light walking (15–30 mins)",
        1: "🧘 Do chair-based stretching & relaxation",
        2: "💪 Try structured low-intensity exercises",
        3: "🍽️ Practice portion control in meals",
        4: "🥗 Eat high-volume, low-calorie foods",
        5: "📊 Track and adjust calorie intake",
        6: "😴 Ensure 7–9 hours of quality sleep",
        7: "🚭 Reduce smoking habits gradually",
        8: "❤️ Monitor blood pressure regularly"
    }
    main_focus = actions_map.get(action, "Stay consistent with your routine")

    # --- Heuristic-based plan ---
    plan = []

    # Morning
    morning = []
    if bmi > 30:
        morning.append("🥗 Start your day with oatmeal + fruit (low GI breakfast)")
    else:
        morning.append("🍳 Balanced breakfast with protein and whole grains")
    if steps < 5000:
        morning.append("🚶 Take a brisk 15-min morning walk")
    plan.append(("🌅 Morning", morning))

    # Afternoon
    afternoon = []
    if stress > 6:
        afternoon.append("🧘 Practice 10 mins of deep breathing or meditation")
    if sys_bp > 130:
        afternoon.append("🧂 Reduce salty snacks during lunch")
    afternoon.append("🥗 Eat a light balanced lunch (veggies + lean protein)")
    plan.append(("🌞 Afternoon", afternoon))

    # Evening
    evening = []
    if fitness < 3:
        evening.append("🏋️ Do bodyweight exercises (push-ups, squats, stretches)")
    else:
        evening.append("🏃 30 mins of moderate activity (cycling, jogging)")
    if sleep < 7:
        evening.append("📵 Limit screen time 1 hour before bed")
    evening.append("🍲 Light dinner (avoid fried/junk food)")
    plan.append(("🌙 Evening", evening))

    # Extra rules
    extras = []
    if smoking == "Current":
        extras.append("🚭 Consider cutting down 1–2 cigarettes/day")
    if heart_rate > 90:
        extras.append("❤️ Your resting heart rate is high – monitor and consult doctor if persistent")
    if dia_bp > 90:
        extras.append("⚠️ Keep track of your blood pressure daily")

    # --- Display Results ---
    st.success(f"✅ Main Focus for Today: **{main_focus}**")

    st.subheader("📅 Personalized Daily Plan")
    for time, tips in plan:
        st.markdown(f"**{time}**")
        for t in tips:
            st.write("- " + t)

    if extras:
        st.subheader("⚠️ Additional Notes")
        for e in extras:
            st.write("- " + e)
