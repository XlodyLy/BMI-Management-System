import streamlit as st
from stable_baselines3 import PPO
from bmi_env import BMIMgmtEnv


# Load trained PPO model
model = PPO.load("models/ppo_bmi.zip")

st.title("🏥 Intelligent BMI Management System")

st.write("Enter your health details below and get a personalized daily recommendation:")

# Input fields
age = st.slider("Age", 18, 80, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.slider("BMI", 15.0, 40.0, 28.0)
sleep = st.slider("Average Sleep (hours)", 4.0, 10.0, 7.0)

# Simplified default values for other factors
exercise = 0.0
diet = 0.0
missed_meals = 0.0
junk_food = 0.0
missed_exercise = 0.0

# Build state vector
state = [age, 0 if gender == "Male" else 1, bmi,
         exercise, diet, sleep, missed_meals, junk_food, missed_exercise]

# Predict recommendation
action, _ = model.predict(state, deterministic=True)
action = int(action)  # convert numpy array to integer


actions_map = {
    0: "Encourage light walking (15–30 mins)",
    1: "Do chair-based stretching",
    2: "Try structured exercises (low intensity)",
    3: "Practice portion control",
    4: "Eat high-volume, low-calorie foods",
    5: "Track and adjust calorie intake",
    6: "Ensure 7–9 hours of sleep"
}

st.subheader("✅ Recommended Action for Today:")
st.success(actions_map[action])
