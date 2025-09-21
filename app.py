import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
DATA_DIR = "data"

# --- Model Training Function ---
def train_and_save_artifacts():
    """
    Loads data, trains the model, and saves the scaler and best model.
    This function will only run if the artifact files don't exist.
    """
    st.info("🚀 First-time setup: Training model and saving artifacts...")
    
    # 1. Load and Clean Data
    try:
        exp1_df = pd.read_csv(os.path.join(DATA_DIR, 'Exp1.csv'), na_values='na')
        exp2_df = pd.read_csv(os.path.join(DATA_DIR, 'Exp2.csv'), na_values='na')
        prep_df = pd.read_csv(os.path.join(DATA_DIR, 'Prep.csv'), na_values='na')
    except FileNotFoundError:
        st.error(f"Error: Data files not found. Ensure your CSV files are in a '{DATA_DIR}' subfolder.")
        return None, None

    exp1_df.rename(columns={'ap(doc)': 'ap', 'vc(cutting speed)': 'vc', 'f(feed rate)': 'f'}, inplace=True)
    full_df = pd.concat([exp1_df, exp2_df, prep_df], ignore_index=True, sort=False)
    relevant_cols = ['ap', 'vc', 'f', 'Fx', 'Fy', 'Fz', 'Tool_ID', 'TCond', 'Ra']
    data = full_df[relevant_cols].copy()
    data.dropna(inplace=True)
    
    # 2. Preprocessing
    X = data.drop('Ra', axis=1)
    y = data['Ra']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. Train the best model (Gradient Boosting was the best in your notebook)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 4. Save the artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    st.success("✅ Model and scaler have been trained and saved locally!")
    return model, scaler

# --- Check for Artifacts and Load/Train ---
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    model, scaler = train_and_save_artifacts()
    if model is None:
        st.stop()
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# --- Streamlit App UI ---
st.set_page_config(page_title="Machining Process Optimizer", page_icon="⚙️", layout="wide")
st.title("⚙️ Surface Roughness ($Ra$) Predictor & Optimizer")

st.sidebar.header("Input Machining Parameters")

def user_input_features():
    ap = st.sidebar.slider('Depth of Cut (ap) [mm]', 0.2, 0.8, 0.5)
    vc = st.sidebar.slider('Cutting Speed (vc) [m/min]', 310.0, 390.0, 350.0)
    f = st.sidebar.slider('Feed Rate (f) [mm/rev]', 0.07, 0.13, 0.1)
    Fx = st.sidebar.slider('Cutting Force X (Fx) [N]', 40.0, 300.0, 137.0)
    Fy = st.sidebar.slider('Cutting Force Y (Fy) [N]', 40.0, 250.0, 93.0)
    Fz = st.sidebar.slider('Cutting Force Z (Fz) [N]', 15.0, 230.0, 88.0)
    Tool_ID = st.sidebar.slider('Tool ID', 10.0, 85.0, 44.0)
    TCond = st.sidebar.slider('Tool Condition (TCond)', 0.0, 0.3, 0.06)
    
    data = {'ap': ap, 'vc': vc, 'f': f, 'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'Tool_ID': Tool_ID, 'TCond': TCond}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Current Input Parameters")
    st.write(input_df.T.rename(columns={0: 'Value'}))

with col2:
    st.subheader("Prediction Result")
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.metric(label="Predicted Surface Roughness (Ra)", value=f"{prediction[0]:.4f} µm")

st.markdown("---")

st.header("🚀 Find Optimal Settings")
if st.button("Optimize Parameters", type="primary"):
    fixed_params = input_df.drop(columns=['ap', 'vc', 'f'])
    
    def objective_function(params):
        ap, vc, f = params
        trial_df = pd.DataFrame([{'ap': ap, 'vc': vc, 'f': f, **fixed_params.iloc[0].to_dict()}])
        trial_scaled = scaler.transform(trial_df)
        return model.predict(trial_scaled)[0]

    bounds = [(0.25, 0.8), (310.0, 390.0), (0.07, 0.13)]

    with st.spinner('Running optimization...'):
        result = differential_evolution(objective_function, bounds, seed=42)
    
    st.success("Optimization Complete!")
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        st.subheader("Optimal Parameters Found:")
        st.info(f"**Depth of Cut (ap):** `{result.x[0]:.4f}` mm")
        st.info(f"**Cutting Speed (vc):** `{result.x[1]:.4f}` m/min")
        st.info(f"**Feed Rate (f):** `{result.x[2]:.4f}` mm/rev")
    with opt_col2:
        st.subheader("Resulting Prediction:")
        st.metric(label="Predicted Minimum Ra", value=f"{result.fun:.4f} µm")