"""
Student Performance Dashboard - Fixed Version
Handles missing columns automatically
Run with: py -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: #333;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .rec-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: #1a1a2e;
        color: white;
        border-radius: 1rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS AND DATA
# ============================================
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/label_encoders.pkl')
        return model, scaler, encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load all data"""
    try:
        df = pd.read_csv('data/student_data_final.csv')
        return df
    except:
        try:
            df = pd.read_csv('data/student_data.csv')
            return df
        except:
            return None

# Load everything
model, scaler, encoders = load_models()
df = load_data()

# Check if data loaded
if df is None:
    st.error("❌ Data not found! Please run data preprocessing first.")
    st.info("1. Open terminal")
    st.info("2. Run: py 01_data_preprocessing.py")
    st.info("3. Then: py model_training.py")
    st.stop()

if model is None:
    st.error("❌ Model not found! Please run model training first.")
    st.info("Run: py model_training.py")
    st.stop()

# ============================================
# GET FEATURE COLUMNS FROM MODEL
# ============================================
# Get the feature columns the model was trained on
try:
    # Try to get feature names from model if available
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
    else:
        # Fallback: use columns from training data (excluding target)
        model_features = [col for col in df.columns if col not in ['passed', 'G3', 'performance_level', 'needs_improvement']]
except:
    model_features = [col for col in df.columns if col not in ['passed', 'G3', 'performance_level', 'needs_improvement']]

# ============================================
# HEADER SECTION
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Performance Prediction System</h1>
    <p>AI-Powered Early Warning System & Personalized Intervention Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "🎯 Predict Student", "💡 Insights & Recommendations"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.markdown(f"**Total Students:** {len(df)}")
st.sidebar.markdown(f"**Pass Rate:** {df['passed'].mean()*100:.1f}%")
st.sidebar.markdown(f"**Avg Grade:** {df['G3'].mean():.1f}/20")

# ============================================
# PAGE 1: DASHBOARD
# ============================================
if page == "📊 Dashboard":
    st.header("📊 Performance Dashboard")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pass_rate = (df['passed'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pass Rate</div>
            <div class="metric-value">{pass_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_grade = df['G3'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Grade</div>
            <div class="metric-value">{avg_grade:.1f}/20</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        at_risk = len(df[df['G3'] < 10])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">At-Risk Students</div>
            <div class="metric-value">{at_risk}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Grade Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['G3'], 
            nbinsx=20, 
            marker_color='#667eea',
            name='Students',
            opacity=0.7
        ))
        fig.add_vline(x=10, line_dash="dash", line_color="red",
                     annotation_text="Pass Threshold")
        fig.update_layout(height=400, showlegend=False,
                         xaxis_title="Final Grade", yaxis_title="Number of Students")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📚 Study Time Impact")
        if 'study_time' in df.columns:
            study_data = df.groupby('study_time')['G3'].mean().reset_index()
            fig = px.bar(study_data, x='study_time', y='G3', color='study_time',
                        title="Average Grade by Study Time")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Study time data not available")
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Failure Impact")
        if 'past_failures' in df.columns:
            fail_data = df.groupby('past_failures')['G3'].mean().reset_index()
            fig = px.line(fail_data, x='past_failures', y='G3', markers=True,
                         title="Grade vs Past Failures")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Past failures data not available")
    
    with col2:
        st.subheader("🎯 Family Support")
        if 'family_support' in df.columns:
            support_data = df.groupby('family_support')['passed'].mean() * 100
            support_data = support_data.reset_index()
            fig = px.bar(support_data, x='family_support', y='passed', color='family_support',
                        title="Pass Rate by Family Support")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Family support data not available")
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    available_cols = [col for col in ['study_time', 'past_failures', 'absences', 'G1', 'G2', 'G3', 'passed'] 
                     if col in df.columns]
    if len(available_cols) >= 2:
        corr_data = df[available_cols].corr()
        fig = px.imshow(corr_data, text_auto=True, aspect="auto",
                       color_continuous_scale='RdBu', title="Correlation Matrix")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2: PREDICT STUDENT
# ============================================
elif page == "🎯 Predict Student":
    st.header("🎯 Real-Time Student Performance Prediction")
    st.markdown("Enter student information below to get instant prediction")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Student Information")
        
        with st.expander("👤 Demographics", expanded=True):
            age = st.slider("Age", 15, 22, 17, help="Student's age in years")
            gender = st.selectbox("Gender", ["M", "F"])
        
        with st.expander("📚 Academic Metrics", expanded=True):
            study_time = st.slider("Weekly Study Time", 1, 4, 2,
                help="1: <2 hours, 2: 2-5 hours, 3: 5-10 hours, 4: >10 hours")
            past_failures = st.slider("Past Failures", 0, 3, 0)
            absences = st.slider("Number of Absences", 0, 40, 5)
            G1 = st.slider("First Period Grade", 0, 20, 12)
            G2 = st.slider("Second Period Grade", 0, 20, 12)
        
        with st.expander("🏠 Support Systems"):
            family_support = st.selectbox("Family Support", ["yes", "no"])
            school_support = st.selectbox("School Support", ["yes", "no"])
            internet = st.selectbox("Internet Access", ["yes", "no"])
            higher_education = st.selectbox("Higher Education Goal", ["yes", "no"])
        
        predict_btn = st.button("🔮 Predict Performance", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        if predict_btn:
            try:
                # Create base input with all required features
                input_dict = {}
                
                # Add all model features with default values
                for feature in model_features:
                    input_dict[feature] = 0
                
                # Override with actual values
                input_dict['age'] = age
                input_dict['study_time'] = study_time
                input_dict['past_failures'] = past_failures
                input_dict['absences'] = absences
                input_dict['G1'] = G1
                input_dict['G2'] = G2
                
                # Handle categorical variables
                input_dict['gender'] = 0 if gender == 'M' else 1
                input_dict['family_support'] = 1 if family_support == 'yes' else 0
                input_dict['school_support'] = 1 if school_support == 'yes' else 0
                input_dict['internet'] = 1 if internet == 'yes' else 0
                input_dict['higher_education'] = 1 if higher_education == 'yes' else 0
                
                # Create DataFrame
                input_data = pd.DataFrame([input_dict])
                
                # Ensure correct column order
                input_data = input_data[model_features]
                
                # Scale features
                input_scaled = scaler.transform(input_data)
                
                # Predict
                proba = model.predict_proba(input_scaled)[0][1]
                
                # Display result
                if proba >= 0.7:
                    st.markdown('<div class="risk-low">', unsafe_allow_html=True)
                    st.markdown("### ✅ LOW RISK STUDENT")
                    st.markdown(f"**Prediction:** Pass")
                    st.markdown(f"**Success Probability:** {proba:.1%}")
                    st.balloons()
                elif proba >= 0.4:
                    st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ MEDIUM RISK STUDENT")
                    st.markdown(f"**Prediction:** {'Pass' if proba >= 0.5 else 'Fail'}")
                    st.markdown(f"**Success Probability:** {proba:.1%}")
                else:
                    st.markdown('<div class="risk-high">', unsafe_allow_html=True)
                    st.markdown("### 🔴 HIGH RISK STUDENT")
                    st.markdown(f"**Prediction:** Fail")
                    st.markdown(f"**Success Probability:** {proba:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={'text': "Success Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ff6b6b"},
                            {'range': [40, 70], 'color': "#ffd43b"},
                            {'range': [70, 100], 'color': "#51cf66"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Personalized Recommendations
                st.markdown("### 💡 Personalized Recommendations")
                
                if proba < 0.4:
                    st.markdown("#### 🚨 Immediate Actions Required:")
                    if study_time < 3:
                        st.write("• 📚 **Increase study time** to at least 5-10 hours per week")
                    if past_failures > 0:
                        st.write("• 👨‍🏫 **Address past failures** through targeted tutoring")
                    if absences > 10:
                        st.write("• 📝 **Improve attendance** - set weekly attendance goals")
                    if family_support == 'no':
                        st.write("• 🗣️ **Engage family support** - schedule parent-teacher meetings")
                    st.write("• 🎯 **Schedule counseling** with academic advisor immediately")
                    
                elif proba < 0.7:
                    st.markdown("#### 📈 Improvement Opportunities:")
                    st.write("• 📖 **Maintain consistent study schedule**")
                    st.write("• 🤝 **Join study groups** for collaborative learning")
                    st.write("• 🎯 **Focus on weak subjects** identified in G1/G2")
                    st.write("• 💻 **Use online resources** for additional practice")
                    
                else:
                    st.markdown("#### 🌟 Maintain Excellence:")
                    st.write("• ✨ **Continue excellent study habits**")
                    st.write("• 🚀 **Consider advanced courses** or honors programs")
                    st.write("• 🌟 **Mentor struggling peers** to reinforce learning")
                    st.write("• 🎯 **Set higher academic goals** for next term")
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please ensure all required fields are filled correctly")

# ============================================
# PAGE 3: INSIGHTS & RECOMMENDATIONS
# ============================================
else:
    st.header("💡 Key Insights & Recommendations")
    st.markdown("Data-driven strategies for student success")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Key Factors Affecting Performance")
        
        impact_data = []
        
        if 'study_time' in df.columns:
            high_study = df[df['study_time'] >= 3]['G3'].mean()
            low_study = df[df['study_time'] <= 2]['G3'].mean()
            impact_data.append({'Factor': 'Study Time', 'Impact': high_study - low_study})
        
        if 'family_support' in df.columns:
            high_family = df[df['family_support'] == 'yes']['G3'].mean()
            low_family = df[df['family_support'] == 'no']['G3'].mean()
            impact_data.append({'Factor': 'Family Support', 'Impact': high_family - low_family})
        
        if 'past_failures' in df.columns:
            no_failures = df[df['past_failures'] == 0]['G3'].mean()
            has_failures = df[df['past_failures'] > 0]['G3'].mean()
            impact_data.append({'Factor': 'No Past Failures', 'Impact': no_failures - has_failures})
        
        if 'internet' in df.columns:
            with_internet = df[df['internet'] == 'yes']['G3'].mean()
            without_internet = df[df['internet'] == 'no']['G3'].mean()
            impact_data.append({'Factor': 'Internet Access', 'Impact': with_internet - without_internet})
        
        if impact_data:
            impact_df = pd.DataFrame(impact_data)
            fig = px.bar(impact_df, x='Factor', y='Impact', color='Factor',
                        title="Impact on Grades (Grade Points Improvement)")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Factor Analysis")
        
        if 'past_failures' in df.columns:
            risk_by_failures = df.groupby('past_failures')['passed'].mean() * 100
            risk_df = risk_by_failures.reset_index()
            fig = px.line(risk_df, x='past_failures', y='passed', markers=True,
                         title="Pass Rate by Past Failures")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        if 'absences' in df.columns:
            df['absences_group'] = pd.cut(df['absences'], bins=[-1, 5, 10, 20, 100], 
                                          labels=['0-5', '6-10', '11-20', '20+'])
            risk_by_absences = df.groupby('absences_group')['passed'].mean() * 100
            risk_abs_df = risk_by_absences.reset_index()
            fig = px.bar(risk_abs_df, x='absences_group', y='passed',
                        title="Pass Rate by Absences")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Actionable Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔴 High Risk Students")
        st.markdown("**Characteristics:**")
        st.write("• Grade < 10")
        st.write("• Multiple past failures")
        st.write("• High absences (>15)")
        st.write("• Low study time (<2h/week)")
        st.markdown("**Recommended Actions:**")
        st.write("✓ Immediate counseling session")
        st.write("✓ Daily progress monitoring")
        st.write("✓ Mandatory tutoring program")
    
    with col2:
        st.markdown("#### 🟡 Medium Risk Students")
        st.markdown("**Characteristics:**")
        st.write("• Grade 10-14")
        st.write("• 1-2 past failures")
        st.write("• Moderate absences (5-15)")
        st.markdown("**Recommended Actions:**")
        st.write("✓ Weekly check-ins with teacher")
        st.write("✓ Study skills workshop")
        st.write("✓ Peer mentoring program")
    
    with col3:
        st.markdown("#### 🟢 Low Risk Students")
        st.markdown("**Characteristics:**")
        st.write("• Grade > 14")
        st.write("• No past failures")
        st.write("• Low absences (<5)")
        st.markdown("**Recommended Actions:**")
        st.write("✓ Maintain current habits")
        st.write("✓ Advanced placement courses")
        st.write("✓ Mentor struggling peers")

# ============================================
# FOOTER
# ============================================
st.markdown("""

""", unsafe_allow_html=True)