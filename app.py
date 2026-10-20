"""
Vehicle Accident Risk Prediction - Streamlit web app (pure Python)
Run locally:  streamlit run app.py
Deploy:       push to GitHub, then connect the repo at share.streamlit.io
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

FEATURES = ["driver_age", "weather", "speed_limit"]
TARGET = "high_risk"
WEATHER = {0: "Fine", 1: "Rain", 2: "Snow", 3: "Fog", 4: "High winds"}
CSV_PATH = "merged_road_safety.csv"   # put the real Kaggle file next to app.py to use it
VIDEO = ("https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/"
         "hf_20260809_012548_ef22562c-c0ae-4816-ad9d-f8922af4e6a7.mp4")

st.set_page_config(page_title="Accident Risk Predictor", page_icon="⚠️", layout="centered")

# ------------------------------------------------------------------ STYLE
st.markdown(f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap");
@import url("https://db.onlinewebfonts.com/c/8cb707a9b8a73f8a7403336b861c3074?family=BubbledotICG-FinePos");
html, body, [class*="css"], .stApp {{font-family:"Inter","Segoe UI",sans-serif; color:#fff;}}
.stApp {{background:#000;}}
header[data-testid="stHeader"], footer, #MainMenu {{visibility:hidden;}}
.bg-video {{position:fixed; inset:0; width:100vw; height:100vh; object-fit:cover; z-index:0; pointer-events:none;}}
.block-container {{position:relative; z-index:1; padding-top:2.5rem; max-width:760px;}}
.badge {{display:inline-block; padding:8px 18px; border-radius:999px; background:#28282a;
         border:1px solid rgba(255,255,255,.4); color:#c4c2c3; font-size:13.5px; font-weight:500;}}
.hero {{text-align:center; margin-bottom:1.2rem;}}
.headline {{font-family:"BubbledotICG-FinePos",monospace; font-size:clamp(30px,6vw,64px);
            letter-spacing:-.04em; line-height:1.12; margin:18px 0 10px; color:#fff;}}
.sub {{color:#d0d0d0; opacity:.8; max-width:520px; margin:0 auto; line-height:1.55;}}
div[data-testid="stTabs"] button {{color:#c8c8c8;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{background:rgba(18,18,20,.72); backdrop-filter:blur(14px);
            border-radius:18px;}}
div.stButton > button {{background:#fff; color:#000; border:0; border-radius:999px; font-weight:600;
            padding:.6rem 1.6rem; box-shadow:0 0 22px rgba(255,255,255,.32);}}
div.stButton > button:hover {{background:#fff; color:#000; transform:translateY(-2px);}}
.result {{font-family:"BubbledotICG-FinePos",monospace; font-size:64px; letter-spacing:-.03em; line-height:1;}}
</style>
<video class="bg-video" autoplay muted loop playsinline><source src="{VIDEO}" type="video/mp4"></video>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ DATA + MODEL
def make_synthetic(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(17, 80, n)
    weather = rng.choice(list(WEATHER), n, p=[0.70, 0.15, 0.04, 0.04, 0.07])
    speed = rng.choice([20, 30, 40, 50, 60, 70], n, p=[0.05, 0.40, 0.12, 0.13, 0.20, 0.10])
    score = (speed - 30) / 20 + np.where(weather > 0, 0.6, 0) + np.where(weather >= 2, 0.5, 0)
    score += np.where(age < 25, 0.7, 0) + np.where(age > 70, 0.5, 0)
    prob = 1 / (1 + np.exp(-(score - 1.6 + rng.normal(0, 0.8, n))))
    return pd.DataFrame({"driver_age": age, "weather": weather, "speed_limit": speed,
                         TARGET: (rng.random(n) < prob).astype(int)})


def load_kaggle(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Age_of_Driver": "driver_age", "Weather_Conditions": "weather",
                            "Speed_limit": "speed_limit"})
    df = df[(df["driver_age"] > 0) & (df["weather"] > 0)]
    wmap = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 4, 7: 3, 8: 0, 9: 0}   # UK codes -> 0-4
    df["weather"] = df["weather"].map(wmap).fillna(0).astype(int)
    df[TARGET] = (df["Accident_Severity"] <= 2).astype(int)
    return df[FEATURES + [TARGET]].dropna()


@st.cache_resource(show_spinner="Training Random Forest…")
def train():
    real = os.path.exists(CSV_PATH)
    df = load_kaggle(CSV_PATH) if real else make_synthetic()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    stats = {"rows": len(df), "real": real, "share": float(y.mean()),
             "acc": accuracy_score(y_te, pred), "prec": precision_score(y_te, pred),
             "rec": recall_score(y_te, pred), "cm": confusion_matrix(y_te, pred)}
    return model, stats


model, stats = train()

# ------------------------------------------------------------------ PAGE
st.markdown("""
<div class="hero">
  <span class="badge">Random Forest · UK road safety data</span>
  <div class="headline">Predict Road Risk<br>Before It Happens</div>
  <p class="sub">Estimate the chance of a serious or fatal accident from driver age, weather and speed limit.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Predictor", "Model performance"])

with tab1:
    with st.container(border=True):
        age = st.slider("Driver age", 17, 85, 35)
        c1, c2 = st.columns(2)
        weather = c1.selectbox("Weather", list(WEATHER), format_func=WEATHER.get)
        speed = c2.selectbox("Speed limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
        if st.button("Predict risk"):
            p = model.predict_proba(pd.DataFrame([[age, weather, speed]], columns=FEATURES))[0, 1]
            high = p >= 0.5
            color = "#ff6b5f" if high else "#4cc38a"
            st.markdown(f'<div class="result" style="color:{color}">{p:.0%}</div>', unsafe_allow_html=True)
            st.markdown(f"**{'High risk' if high else 'Low risk'}**")
            st.progress(float(p))

with tab2:
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{stats['acc']:.1%}")
        m2.metric("Precision", f"{stats['prec']:.1%}")
        m3.metric("Recall", f"{stats['rec']:.1%}")
        st.caption(f"{stats['rows']:,} records · {stats['share']:.1%} high risk · "
                   + ("real Kaggle data" if stats["real"] else "synthetic demo data"))
        st.markdown("**Feature importance**")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        st.bar_chart(imp)
        st.markdown("**Confusion matrix**")
        st.dataframe(pd.DataFrame(stats["cm"], index=["Actual low", "Actual high"],
                                  columns=["Pred low", "Pred high"]))
