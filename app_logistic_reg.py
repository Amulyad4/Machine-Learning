import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Logistic Regression",
    layout="centered"
)
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style2.css")
st.markdown("""
<div class="card">
    <h1>🎄 Logistic Regression 🎄</h1>
    <p>
        Predict whether a customer is a <b>Big Tipper</b> 🎁  
        using Logistic Regression
    </p>
</div>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()
df["big_tip"] = (df["tip"] >= 3).astype(int)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

df_encoded = df.copy()
df_encoded["sex"] = df_encoded["sex"].map({"Male": 1, "Female": 0})
df_encoded["smoker"] = df_encoded["smoker"].map({"Yes": 1, "No": 0})
df_encoded["time"] = df_encoded["time"].map({"Lunch": 0, "Dinner": 1})
df_encoded["day"] = df_encoded["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})

X = df_encoded[["total_bill", "size", "sex", "smoker", "day", "time"]]
y = df_encoded["big_tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎅 Confusion Matrix")

fig, ax = plt.subplots()
ax.imshow(cm, cmap="Greens")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Normal", "Big"])
ax.set_yticklabels(["Normal", "Big"])

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎄 Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("Error Rate", f"{1-accuracy:.2f}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="card">
    <h3>🎁 Model Coefficients</h3>
    <p>
        <b>Total Bill:</b> {model.coef_[0][0]:.3f}<br>
        <b>Size:</b> {model.coef_[0][1]:.3f}<br>
        <b>Sex:</b> {model.coef_[0][2]:.3f}<br>
        <b>Smoker:</b> {model.coef_[0][3]:.3f}<br>
        <b>Day:</b> {model.coef_[0][4]:.3f}<br>
        <b>Time:</b> {model.coef_[0][5]:.3f}<br><br>
        <b>Intercept:</b> {model.intercept_[0]:.3f}
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Predict Big Tip")

bill = st.slider("💰 Total Bill ($)", float(df.total_bill.min()), float(df.total_bill.max()), 30.0)
size = st.slider("👥 Party Size", 1, 6, 2)
sex = st.selectbox("🧑 Sex", ["Male", "Female"])
smoker = st.selectbox("🚬 Smoker", ["Yes", "No"])
day = st.selectbox("📅 Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("⏰ Time", ["Lunch", "Dinner"])

input_data = np.array([[
    bill,
    size,
    1 if sex == "Male" else 0,
    1 if smoker == "Yes" else 0,
    {"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3}[day],
    0 if time == "Lunch" else 1
]])

input_scaled = scaler.transform(input_data)
prob = model.predict_proba(input_scaled)[0][1]
prediction = model.predict(input_scaled)[0]

result = "🎁 Big Tipper!" if prediction == 1 else "🙂 Normal Tipper"

st.markdown(
    f"""
    <div class="prediction-box">
        {result}<br>
        Probability: {prob:.2%}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
