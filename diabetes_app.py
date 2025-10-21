import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 32px !important;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 24px !important;
        font-weight: bold;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 16px;
        color: #7f8c8d;
        text-align: justify;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive-result {
        background-color: #ffebee;
        border-left: 5px solid #e53935;
        color: #000000;
    }
    .negative-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        color: #000000;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🩺 Diabetes Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">This application predicts the likelihood of diabetes based on medical parameters. Enter patient information below to get a risk assessment.</p>', unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    # Try to load the tuned model first, fallback to regular model
    try:
        with open('best_model_tuned.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    st.success("✅ Machine Learning Model Loaded Successfully!")
except FileNotFoundError:
    st.error("❌ Model files not found. Please run the training script first.")
    st.stop()

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.selectbox("Choose the mode", ["🔍 Prediction", "📊 Data Visualization", "ℹ️ About"])

if app_mode == "🔍 Prediction":
    # Create input fields for all features
    st.markdown('<p class="sub-header">Patient Information</p>', unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📋 Manual Input", "📊 Slider Input"])
    
    with tab1:
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information**")
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1, help="Number of times pregnant")
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100, step=1, help="Plasma glucose concentration")
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1, help="Diastolic blood pressure")
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1, help="Triceps skin fold thickness")
        
        with col2:
            st.markdown("**Advanced Metrics**")
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80, step=1, help="2-Hour serum insulin")
            bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Body mass index")
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5, step=0.01, help="Diabetes pedigree function")
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=25, step=1, help="Age in years")
    
    with tab2:
        # Slider inputs for more interactive experience
        st.markdown("**Adjust values using sliders**")
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies_slider = st.slider("Number of Pregnancies", 0, 20, 0)
            glucose_slider = st.slider("Glucose Level (mg/dL)", 0, 300, 100)
            blood_pressure_slider = st.slider("Blood Pressure (mm Hg)", 0, 200, 70)
            skin_thickness_slider = st.slider("Skin Thickness (mm)", 0, 100, 20)
        
        with col2:
            insulin_slider = st.slider("Insulin Level (mu U/ml)", 0, 1000, 80)
            bmi_slider = st.slider("BMI (Body Mass Index)", 0.0, 100.0, 25.0, 0.1)
            diabetes_pedigree_slider = st.slider("Diabetes Pedigree Function", 0.0, 5.0, 0.5, 0.01)
            age_slider = st.slider("Age (years)", 0, 120, 25)
        
        # Use slider values
        pregnancies = pregnancies_slider
        glucose = glucose_slider
        blood_pressure = blood_pressure_slider
        skin_thickness = skin_thickness_slider
        insulin = insulin_slider
        bmi = bmi_slider
        diabetes_pedigree = diabetes_pedigree_slider
        age = age_slider
    
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Display the input data
    st.markdown('<p class="sub-header">Patient Data Summary</p>', unsafe_allow_html=True)
    st.dataframe(input_data.style.highlight_max(axis=0))
    
    # Add a prediction button with confirmation
    if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
        # Show loading spinner
        with st.spinner("Analyzing patient data..."):
            import time
            time.sleep(1)  # Simulate processing time
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction (suppress warning by converting to numpy array without column names)
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Display results
            st.markdown('<p class="sub-header">Prediction Results</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Risk Level", f"{prediction_proba[0][1]*100:.1f}%", 
                         delta=f"{'High' if prediction_proba[0][1] > 0.5 else 'Low'} Risk")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{max(prediction_proba[0])*100:.1f}%", 
                         delta="Model Confidence")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Model Used", "Random Forest", 
                         delta="Best Performing")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed result display
            if prediction[0] == 1:
                st.markdown(f'''
                <div class="result-box positive-result">
                    <h3 style="color: #000000;">⚠️ Diabetes Risk Detected</h3>
                    <p style="color: #000000;">The model predicts that this patient has <strong>diabetes</strong> with a probability of <strong>{prediction_proba[0][1]*100:.1f}%</strong>.</p>
                    <p style="color: #000000;"><strong>Recommendation:</strong> Consult with a healthcare professional for further evaluation and treatment.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="result-box negative-result">
                    <h3 style="color: #000000;">✅ Low Diabetes Risk</h3>
                    <p style="color: #000000;">The model predicts that this patient <strong>does not have diabetes</strong> with a probability of <strong>{prediction_proba[0][0]*100:.1f}%</strong>.</p>
                    <p style="color: #000000;"><strong>Recommendation:</strong> Continue maintaining a healthy lifestyle and regular check-ups.</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Create probability visualization using Streamlit's built-in progress bar
            st.markdown('<p class="sub-header">Risk Probability Distribution</p>', unsafe_allow_html=True)
            
            # Display probabilities as metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("No Diabetes Probability", f"{prediction_proba[0][0]*100:.1f}%")
            with col2:
                st.metric("Diabetes Probability", f"{prediction_proba[0][1]*100:.1f}%")
            
            # Visualize with progress bars
            st.markdown("**No Diabetes Probability:**")
            st.progress(prediction_proba[0][0], text=f"{prediction_proba[0][0]*100:.1f}%")
            
            st.markdown("**Diabetes Probability:**")
            st.progress(prediction_proba[0][1], text=f"{prediction_proba[0][1]*100:.1f}%")
            
            # Risk level indicator
            st.markdown('<p class="sub-header">Risk Level Indicator</p>', unsafe_allow_html=True)
            risk_level = prediction_proba[0][1]
            if risk_level < 0.3:
                st.info("🟢 Low Risk - Maintain healthy lifestyle")
            elif risk_level < 0.7:
                st.warning("🟡 Moderate Risk - Consider medical consultation")
            else:
                st.error("🔴 High Risk - Immediate medical consultation recommended")

elif app_mode == "📊 Data Visualization":
    st.markdown('<p class="sub-header">📊 Model Performance Visualization</p>', unsafe_allow_html=True)
    
    try:
        # Display the model comparison image
        st.image("model_comparison.png", caption="Model Performance Comparison", use_column_width=True)
        
        # Add explanation
        st.markdown("""
        ### Model Performance Metrics
        - **Accuracy**: Proportion of correct predictions
        - **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives
        - **Feature Importance**: Which features contribute most to predictions
        
        The Random Forest model was selected as the best performer with an accuracy of 75.97%.
        """)
    except FileNotFoundError:
        st.warning("Visualization file not found. Please run the training script to generate it.")

else:  # About section
    st.markdown('<p class="sub-header">ℹ️ About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This application predicts the likelihood of diabetes based on medical parameters using machine learning algorithms.
    
    ### 🧠 Technology Stack
    - **Machine Learning**: Scikit-learn with Random Forest algorithm
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn
    - **Web Interface**: Streamlit
    
    ### 📊 Features
    - **Multiple Input Methods**: Manual entry or slider controls
    - **Real-time Predictions**: Instant results based on input data
    - **Visual Feedback**: Charts and color-coded results
    - **Risk Assessment**: Detailed probability analysis
    
    ### 📋 Data Requirements
    The model requires the following medical parameters:
    1. Number of Pregnancies
    2. Glucose Level (mg/dL)
    3. Blood Pressure (mm Hg)
    4. Skin Thickness (mm)
    5. Insulin Level (mu U/ml)
    6. BMI (Body Mass Index)
    7. Diabetes Pedigree Function
    8. Age (years)
    
    ### ⚠️ Disclaimer
    This tool is for educational purposes only and should not replace professional medical advice.
    Always consult with healthcare professionals for accurate diagnosis and treatment.
    """)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit | Diabetes Prediction Application")