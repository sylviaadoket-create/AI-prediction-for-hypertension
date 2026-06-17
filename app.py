import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPredict AI Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CACHED RESOURCE LOADING (For Fast Performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    models = {
        'Logistic Regression': joblib.load('model_logistic_regression.pkl'),
        'Random Forest': joblib.load('model_random_forest.pkl'),
        'XGBoost': joblib.load('model_xgboost.pkl')
    }
    preprocessor = joblib.load('preprocessor.pkl')
    feature_names = joblib.load('feature_names.pkl')
    metrics_df = pd.read_csv('model_metrics.csv')
    
    # --- FIX APPLIED HERE ---
    # Load and clean raw data to match training preprocessing
    df_raw = pd.read_csv('Heart.csv')
    df_raw = df_raw.replace('?', np.nan)
    
    # Impute missing values exactly as done in the training script
    df_raw['Ca'] = pd.to_numeric(df_raw['Ca'], errors='coerce').fillna(df_raw['Ca'].mode()[0])
    df_raw['Thal'] = df_raw['Thal'].fillna(df_raw['Thal'].mode()[0])
    # ------------------------
    
    return models, preprocessor, feature_names, metrics_df, df_raw

models, preprocessor, feature_names, metrics_df, df_raw = load_assets()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("❤️ CardioPredict AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Dataset Overview", "📈 Model Performance", "🩺 Patient Prediction", "🔍 Explainable AI (XAI)", "ℹ️ About"]
)
st.sidebar.markdown("---")
st.sidebar.info("Built for clinical decision support.")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_recommendation(prob):
    percentage = prob * 100
    if percentage <= 30:
        return "🟢 **Low Risk (0-30%)**: Routine annual checkup recommended. Maintain healthy lifestyle."
    elif 31 <= percentage <= 50:
        return "🟡 **Moderate Risk (31-50%)**: Lifestyle modifications strongly advised. Monitor BP and cholesterol every 3-6 months."
    elif 51 <= percentage <= 70:
        return "🟠 **High Risk (51-70%)**: Consult a cardiologist. Further diagnostic testing is recommended."
    else:
        return "🔴 **Critical Risk (71-100%)**: Immediate medical evaluation required. High probability of cardiovascular event."

# -----------------------------------------------------------------------------
# PAGE: HOME
# -----------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("❤️ CardioPredict AI Dashboard")
    st.markdown("Welcome to the clinical decision support system. This dashboard leverages Machine Learning and Explainable AI (XAI) to assess heart disease risk.")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients in Dataset", len(df_raw))
    col2.metric("Available Models", len(models))
    col3.metric("Best ROC-AUC", f"{metrics_df['ROC-AUC'].max():.2%}")

# -----------------------------------------------------------------------------
# PAGE: DATASET OVERVIEW
# -----------------------------------------------------------------------------
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("Snapshot of the clinical data used to train the models (post-imputation).")
    st.dataframe(df_raw.head(10), use_container_width=True)
    
    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='HD', data=df_raw, palette='viridis', ax=ax)
    ax.set_xticklabels(['No Heart Disease (0)', 'Heart Disease (1)'])
    ax.set_ylabel('Count')
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")
    st.write("Comparison of the pre-trained models on the hold-out test set.")
    
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='lightgreen'), use_container_width=True)
    
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sample from the now-cleaned df_raw
    X_sample = df_raw.drop(columns=['AHD', 'HD']).sample(200, random_state=42)
    y_sample = df_raw.loc[X_sample.index, 'HD']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (name, model) in enumerate(models.items()):
        y_pred_proba = model.predict_proba(X_sample)[:, 1]
        fpr, tpr, _ = roc_curve(y_sample, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# PAGE: PATIENT PREDICTION
# -----------------------------------------------------------------------------
elif page == "🩺 Patient Prediction":
    st.title("🩺 Patient Risk Prediction")
    st.markdown("Enter the patient's clinical details below to generate a risk assessment.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 20, 80, 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
            chest_pain = st.selectbox("Chest Pain Type", ["typical", "asymptomatic", "nonanginal", "nontypical"])
            rest_bp = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            rest_ecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal (0)", "ST-T wave abnormality (1)", "Left ventricular hypertrophy (2)"][x])
        
        with col2:
            max_hr = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
            ex_ang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3], format_func=lambda x: ["Upsloping (1)", "Flat (2)", "Downsloping (3)"][x-1])
            ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", ["normal", "fixed", "reversable"])
            
        model_choice = st.selectbox("Select Prediction Model", list(models.keys()))
        submit_button = st.form_submit_button("🔮 Predict Risk")

    if submit_button:
        input_data = pd.DataFrame({
            'Age': [age], 'Sex': [sex], 'ChestPain': [chest_pain], 'RestBP': [rest_bp],
            'Chol': [chol], 'Fbs': [fbs], 'RestECG': [rest_ecg], 'MaxHR': [max_hr],
            'ExAng': [ex_ang], 'Oldpeak': [oldpeak], 'Slope': [slope], 'Ca': [ca], 'Thal': [thal]
        })
        
        # Ensure column order matches training
        input_data = input_data[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']]
        
        model = models[model_choice]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Prediction", "Heart Disease Detected 🚨" if prediction == 1 else "No Heart Disease ✅")
            st.metric("Risk Probability", f"{probability:.1%}")
            
        with col2:
            st.subheader("📋 Clinical Recommendation")
            st.info(get_recommendation(probability))
            
        # Store for XAI page
        st.session_state['last_input'] = input_data
        st.session_state['last_model'] = model_choice
        st.session_state['last_prob'] = probability

# -----------------------------------------------------------------------------
# PAGE: EXPLAINABLE AI (XAI)
# -----------------------------------------------------------------------------
elif page == "🔍 Explainable AI (XAI)":
    st.title("🔍 Explainable AI (SHAP)")
    st.markdown("Understand *why* the model made its prediction using SHAP.")
    
    if 'last_input' not in st.session_state:
        st.warning("⚠️ Please go to the **Patient Prediction** tab and make a prediction first to see the XAI explanation.")
    else:
        input_data = st.session_state['last_input']
        model_choice = st.session_state['last_model']
        model = models[model_choice]
        
        st.subheader(f"Model: {model_choice}")
        
        # Sample from the now-cleaned df_raw
        background_data = df_raw.drop(columns=['AHD', 'HD']).sample(100, random_state=42)
        background_transformed = preprocessor.transform(background_data)
        input_transformed = preprocessor.transform(input_data)
        
        st.markdown("---")
        st.subheader("1. Global Feature Importance")
        
        if "Random Forest" in model_choice or "XGBoost" in model_choice:
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
        else:
            explainer = shap.LinearExplainer(model.named_steps['classifier'], background_transformed)
            
        shap_values = explainer.shap_values(background_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, background_transformed, feature_names=feature_names, show=False)
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("2. Local Prediction Explanation (Waterfall Plot)")
        
        input_shap_values = explainer.shap_values(input_transformed)
        if isinstance(input_shap_values, list):
            input_shap_values = input_shap_values[1]
            
        explanation = shap.Explanation(
            values=input_shap_values[0],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
            data=input_transformed[0],
            feature_names=feature_names
        )
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig2)

# -----------------------------------------------------------------------------
# PAGE: ABOUT
# -----------------------------------------------------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Dashboard")
    st.markdown("""
    ### Project Overview
    This dashboard demonstrates the deployment of machine learning models for clinical decision support.
    
    ### Recommendations Logic
    - **0% - 30%**: Low Risk
    - **31% - 50%**: Moderate Risk
    - **51% - 70%**: High Risk
    - **71% - 100%**: Critical Risk
    
    ⚠️ **Disclaimer**: This tool is for educational purposes only and should not replace professional medical judgment.
    """)