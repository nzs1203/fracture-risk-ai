import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from openai import OpenAI

# 1. 页面设置
st.set_page_config(page_title="骨折风险智能预测系统", layout="wide")
st.title("🏥 骨折风险多模态 AI 辅助决策系统")
st.markdown("本系统融合 CatBoost 传统机器学习高精度算力与 ChatGPT 临床语义解析能力。")

# 2. 侧边栏：输入患者指标
st.sidebar.header("输入患者检验指标")
age = st.sidebar.slider("年龄 (Age)", 60, 100, 75)
rbc = st.sidebar.slider("红细胞计数 (RBC)", 2.0, 6.0, 3.5)
hb = st.sidebar.slider("血红蛋白 (Hb)", 6.0, 16.0, 10.5)
glu = st.sidebar.slider("空腹血糖 (GLU)", 70.0, 250.0, 130.0)

# 3. 加载模型与API
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('catboost_fracture_model.cbm')
    return model

model = load_model()
# 获取存放在 Streamlit 后台的安全密钥
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

if st.button("🚀 开始智能评估"):
    with st.spinner("AI 正在计算风险并生成报告..."):
        
        # --- 模块 A: CatBoost 精准计算 ---
        input_data = [age, rbc, hb, glu]
        # 假设预测正类(1)的概率在索引 1
        risk_prob = model.predict_proba(input_data)[1] * 100 
        
        st.subheader("📊 CatBoost 精准预测引擎")
        if risk_prob > 50:
            st.error(f"高危预警：骨折发生概率 {risk_prob:.1f}%")
        else:
            st.success(f"低危状态：骨折发生概率 {risk_prob:.1f}%")

        # --- 模块 B: LLM 智能解读 ---
        st.subheader("🤖 大模型智能临床解读")
        
        # 构建给微调大模型的 Prompt
        prompt = f"""
        患者年龄 {age} 岁，RBC {rbc}，Hb {hb}，血糖 {glu}。
        CatBoost计算出的客观骨折风险概率为 {risk_prob:.1f}%。
        请结合上述指标，给出一份简短、专业的临床风险解读，并给出防跌倒或干预建议。
        """
        
        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:niu:bone-model:DaephI7x", # 换成你微调后模型的专属 ID！
            messages=[
                {"role": "system", "content": "你是一位资深的急诊骨科主任医师。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        st.write(response.choices[0].message.content)
