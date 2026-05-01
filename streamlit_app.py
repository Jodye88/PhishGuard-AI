import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    with open("rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    ann = tf.keras.models.load_model("ann_model.h5")
    return rf, ann, scaler, features

rf_model, ann_model, scaler, feature_names = load_assets()

# --- UI ---
st.set_page_config(page_title="PhishGuard AI", layout="wide")
st.title("🛡️ PhishGuard AI: Production Security Suite")

# --- SCANNER TAB ---
st.subheader("Manual Website Analysis")
with st.form("scanner_form"):
    cols = st.columns(4)
    user_inputs = []
    for i, name in enumerate(feature_names):
        val = cols[i % 4].selectbox(name, [1, 0, -1], index=0)
        user_inputs.append(val)
    
    submitted = st.form_submit_button("Run Dual-Model Analysis")

if submitted:
    data = np.array(user_inputs).reshape(1, -1)
    scaled_data = scaler.transform(data)
    
    rf_res = rf_model.predict(scaled_data)[0]
    ann_res = (ann_model.predict(scaled_data, verbose=0) > 0.5).astype(int)[0][0]
    
    c1, c2 = st.columns(2)
    c1.metric("Random Forest", "SAFE" if rf_res == 1 else "PHISHING")
    c2.metric("ANN", "SAFE" if ann_res == 1 else "PHISHING")
    
    if rf_res == 1 and ann_res == 1:
        st.success("✅ CONSENSUS: SITE IS SECURE")
    elif rf_res == 0 and ann_res == 0:
        st.error("🚨 CONSENSUS: PHISHING DETECTED")
    else:
        st.warning("⚠️ CONFLICT: MODELS DISAGREE (Medium Risk)")