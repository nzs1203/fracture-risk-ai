import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from openai import OpenAI

# 1. Page Configuration (Minimalist Academic Style)
st.set_page_config(page_title="Fracture Risk CDSS", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a cleaner, academic look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #2c3e50; color: white; border-radius: 4px;}
    h1, h2, h3 {color: #2c3e50; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;}
    .report-box {padding: 20px; background-color: #ffffff; border-left: 5px solid #34495e; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

st.title("Multimodal AI Clinical Decision Support System")
st.markdown("**Objective:** Pathological Fracture Risk Stratification in Hospitalized Osteoporosis Patients.")
st.markdown("---")

# 2. Sidebar: Clinical Parameters Input
st.sidebar.header("Clinical Parameters")
st.sidebar.markdown("Please input patient laboratory data:")

age = st.sidebar.slider("Age (years)", 60, 100, 75)
rbc = st.sidebar.slider("RBC (x10¹²/L)", 2.0, 6.0, 3.5, step=0.1)
hb = st.sidebar.slider("Hemoglobin (g/dL)", 6.0, 16.0, 10.5, step=0.1)
glu = st.sidebar.slider("Fasting Glucose (mg/dL)", 70.0, 250.0, 130.0, step=1.0)

# 3. Load ML Model and API
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    # Ensure your .cbm file is in the same directory
    model.load_model('catboost_fracture_model.cbm') 
    return model

model = load_model()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

# 4. Assessment Execution
if st.sidebar.button("Run AI Assessment"):
    with st.spinner("Executing risk stratification and generating clinical interpretation..."):
        
        # --- Module A: CatBoost Machine Learning Engine ---
        input_data = [age, rbc, hb, glu]
        risk_prob = model.predict_proba(input_data)[1] * 100 
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("I. ML Risk Stratification")
            st.markdown("**(CatBoost Algorithm)**")
            
            # Use academic metric display instead of colorful success/error boxes
            st.metric(label="Predicted Fracture Probability", value=f"{risk_prob:.2f}%")
            
            if risk_prob > 50:
                st.warning("Status: High Risk Profile")
            else:
                st.info("Status: Low/Moderate Risk Profile")

        # --- Module B: LLM Clinical Interpretation ---
        with col2:
            st.subheader("II. LLM Clinical Interpretation")
            st.markdown("**(Fine-tuned Generative Model)**")
            
            # Highly formal English medical prompt
            prompt = f"""
            Patient Profile: 
            - Age: {age} years
            - Red Blood Cell Count (RBC): {rbc} x10^12/L
            - Hemoglobin (Hb): {hb} g/dL
            - Fasting Blood Glucose (GLU): {glu} mg/dL
            
            The CatBoost machine learning algorithm has computed a pathological fracture risk probability of {risk_prob:.2f}%.
            
            Task: Provide a concise, evidence-based clinical interpretation of this risk. Analyze the potential synergistic effects of the patient's anemia (if present) and glycemic status on fall risk and skeletal vulnerability. Conclude with actionable clinical recommendations (e.g., DXA scanning, fall prevention strategies). Use formal, academic medical terminology.
            """
            
            response = client.chat.completions.create(
                model="ft:gpt-4o-mini-2024-07-18:niu:bone-model:DaephI7x", # 别忘了换成你自己的微调模型 ID
                messages=[
                    {"role": "system", "content": "You are a senior attending physician specializing in emergency medicine and geriatrics, providing expert clinical decision support."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Display report in a clean, academic box
            st.markdown(f'<div class="report-box">{response.choices[0].message.content}</div>', unsafe_allow_html=True)
            
else:
    st.info("Awaiting input. Please adjust parameters in the sidebar and click 'Run AI Assessment'.")
