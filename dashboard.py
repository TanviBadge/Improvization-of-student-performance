"""
Student Performance Dashboard - With Top Tab Navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Student Performance Dashboard", page_icon="🎓", layout="wide")

# ============================================
# AUTO-SETUP (same as before)
# ============================================
def setup_if_needed():
    data_exists = os.path.exists('data/student_data.csv')
    model_exists = os.path.exists('models/best_model.pkl')
    
    if not data_exists or not model_exists:
        with st.spinner("⚙️ First-time setup..."):
            os.makedirs('data', exist_ok=True)
            os.makedirs('models', exist_ok=True)
            
            if not data_exists:
                np.random.seed(42)
                n = 1000
                data = {
                    'age': np.random.randint(15, 23, n),
                    'gender': np.random.choice(['M', 'F'], n),
                    'study_time': np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.35, 0.3, 0.15]),
                    'past_failures': np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05]),
                    'absences': np.random.randint(0, 40, n),
                    'family_support': np.random.choice(['yes', 'no'], n, p=[0.55, 0.45]),
                    'school_support': np.random.choice(['yes', 'no'], n, p=[0.25, 0.75]),
                    'internet': np.random.choice(['yes', 'no'], n, p=[0.65, 0.35]),
                    'higher_education': np.random.choice(['yes', 'no'], n, p=[0.7, 0.3]),
                    'G1': np.zeros(n), 'G2': np.zeros(n), 'G3': np.zeros(n)
                }
                for i in range(n):
                    base = data['study_time'][i] * 3.5
                    base -= data['past_failures'][i] * 2.5
                    base -= data['absences'][i] * 0.1
                    if data['family_support'][i] == 'yes': base += 1.2
                    if data['school_support'][i] == 'yes': base += 0.8
                    if data['higher_education'][i] == 'yes': base += 1.5
                    base += np.random.normal(0, 2)
                    final = max(0, min(20, base))
                    data['G3'][i] = final
                    data['G2'][i] = max(0, min(20, final - np.random.randint(0, 3)))
                    data['G1'][i] = max(0, min(20, final - np.random.randint(0, 4)))
                df = pd.DataFrame(data)
                df['passed'] = (df['G3'] >= 10).astype(int)
                df.to_csv('data/student_data.csv', index=False)
            
            if not model_exists:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.ensemble import GradientBoostingClassifier
                
                df = pd.read_csv('data/student_data.csv')
                X = df.drop(columns=['passed', 'G3'])
                y = df['passed']
                
                encoders = {}
                for col in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le
                joblib.dump(encoders, 'models/label_encoders.pkl')
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                joblib.dump(scaler, 'models/scaler.pkl')
                
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_s, y_train)
                joblib.dump(model, 'models/best_model.pkl')

setup_if_needed()

# ============================================
# LOAD
# ============================================
@st.cache_resource
def load_models():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    return model, scaler, encoders

@st.cache_data
def load_data():
    return pd.read_csv('data/student_data.csv')

model, scaler, encoders = load_models()
df = load_data()

# ============================================
# HEADER
# ============================================
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;">
    <h1 style="margin: 0;">🎓 Student Performance Prediction System</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">AI-Powered Early Warning System</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# TOP TABS NAVIGATION
# ============================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Predict Student", "💡 Insights"])

# ============================================
# TAB 1: DASHBOARD
# ============================================
with tab1:
    st.header("📊 Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Students", len(df))
    with col2: st.metric("Pass Rate", f"{(df['passed'].mean()*100):.1f}%")
    with col3: st.metric("Average Grade", f"{df['G3'].mean():.1f}/20")
    with col4: st.metric("At-Risk Students", len(df[df['G3'] < 10]))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Grade Distribution")
        fig = px.histogram(df, x='G3', nbins=20, color='passed',
                          color_discrete_map={1: '#51cf66', 0: '#ff6b6b'})
        fig.add_vline(x=10, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📚 Study Time Impact")
        study_data = df.groupby('study_time')['G3'].mean().reset_index()
        fig = px.bar(study_data, x='study_time', y='G3', title="Average Grade by Study Time")
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2: PREDICT
# ============================================
with tab2:
    st.header("🎯 Predict Student Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 15, 22, 17)
        gender = st.selectbox("Gender", ["M", "F"])
        study_time = st.slider("Study Time", 1, 4, 2)
        past_failures = st.slider("Past Failures", 0, 3, 0)
        absences = st.slider("Absences", 0, 40, 5)
        G1 = st.slider("First Period Grade", 0, 20, 12)
        G2 = st.slider("Second Period Grade", 0, 20, 12)
        family_support = st.selectbox("Family Support", ["yes", "no"])
        
        predict_btn = st.button("🔮 Predict Performance", type="primary")
    
    with col2:
        if predict_btn:
            input_dict = {}
            for col in df.columns:
                if col not in ['passed', 'G3']:
                    input_dict[col] = 0
            
            input_dict['age'] = age
            input_dict['study_time'] = study_time
            input_dict['past_failures'] = past_failures
            input_dict['absences'] = absences
            input_dict['G1'] = G1
            input_dict['G2'] = G2
            input_dict['gender'] = 0 if gender == 'M' else 1
            input_dict['family_support'] = 1 if family_support == 'yes' else 0
            
            input_df = pd.DataFrame([input_dict])
            feature_cols = [c for c in df.columns if c not in ['passed', 'G3']]
            input_df = input_df[feature_cols]
            
            input_scaled = scaler.transform(input_df)
            proba = model.predict_proba(input_scaled)[0][1]
            
            if proba >= 0.7:
                st.success(f"### ✅ LOW RISK\n**Success Probability:** {proba:.1%}")
            elif proba >= 0.4:
                st.warning(f"### ⚠️ MEDIUM RISK\n**Success Probability:** {proba:.1%}")
            else:
                st.error(f"### 🔴 HIGH RISK\n**Success Probability:** {proba:.1%}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                gauge={'axis': {'range': [0, 100]},
                       'steps': [{'range': [0, 40], 'color': "#ff6b6b"},
                                {'range': [40, 70], 'color': "#ffd43b"},
                                {'range': [70, 100], 'color': "#51cf66"}]}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: INSIGHTS
# ============================================
with tab3:
    st.header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        study_impact = df.groupby('study_time')['passed'].mean() * 100
        fig = px.line(x=study_impact.index, y=study_impact.values, markers=True,
                     title="Success Rate by Study Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fail_impact = df.groupby('past_failures')['passed'].mean() * 100
        fig = px.bar(x=fail_impact.index, y=fail_impact.values,
                    title="Success Rate by Past Failures")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("🚀 **Student Performance Analytics** | Powered by Machine Learning")
