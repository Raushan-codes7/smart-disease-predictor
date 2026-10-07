"""
Streamlit web application for disease prediction system.
Interactive UI for symptom input and disease prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_training import predict_disease, get_feature_importance


# Page configuration
st.set_page_config(
    page_title="Smart Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .symptom-tag {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model_components():
    """Load model and related components with caching."""
    try:
        model_dir = 'models'
        
        # Load model
        with open(f'{model_dir}/disease_predictor.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load label encoder
        with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load feature names
        with open(f'{model_dir}/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load performance metrics
        with open(f'{model_dir}/performance_metrics.pkl', 'rb') as f:
            performance = pickle.load(f)
        
        return model, label_encoder, feature_names, performance
    
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run the training pipeline first by executing 'python main.py'")
        return None, None, None, None


def display_header():
    """Display the main header and description."""
    st.markdown('<h1 class="main-header">🏥 Smart Disease Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Enter your symptoms below to get an AI-powered disease prediction with confidence scores.
        </p>
        <p style="font-size: 1rem; color: #888;">
            <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar_info(performance, feature_names):
    """Display model information in sidebar."""
    st.sidebar.markdown("## 📊 Model Information")
    
    if performance:
        # Model performance metrics
        st.sidebar.markdown("### Model Performance")
        accuracy = performance['accuracy']
        st.sidebar.metric("Accuracy", f"{accuracy:.1%}")
        
        # Display other metrics if available
        if 'classification_report' in performance:
            report = performance['classification_report']
            if 'macro avg' in report:
                st.sidebar.metric("Precision (Macro)", f"{report['macro avg']['precision']:.3f}")
                st.sidebar.metric("Recall (Macro)", f"{report['macro avg']['recall']:.3f}")
                st.sidebar.metric("F1-Score (Macro)", f"{report['macro avg']['f1-score']:.3f}")
    
    # Dataset information
    st.sidebar.markdown("### Dataset Information")
    st.sidebar.write(f"**Total Symptoms:** {len(feature_names)}")
    
    if performance and 'classification_report' in performance:
        num_classes = len([k for k in performance['classification_report'].keys() 
                          if k not in ['accuracy', 'macro avg', 'weighted avg']])
        st.sidebar.write(f"**Disease Classes:** {num_classes}")
    
    # Instructions
    st.sidebar.markdown("### 📝 Instructions")
    st.sidebar.markdown("""
    1. Select your symptoms from the dropdown
    2. Click "Predict Disease" 
    3. View predictions with confidence scores
    4. Check detailed analysis below
    """)


def create_symptom_input(feature_names):
    """Create symptom input interface."""
    st.markdown("## 🔍 Symptom Input")
    
    # Multi-select for symptoms
    selected_symptoms = st.multiselect(
        "Select your symptoms:",
        options=feature_names,
        default=[],
        help="Choose all symptoms that apply to you"
    )
    
    # Display selected symptoms
    if selected_symptoms:
        st.markdown("**Selected Symptoms:**")
        symptom_tags = " ".join([f'<span class="symptom-tag">{symptom}</span>' 
                               for symptom in selected_symptoms])
        st.markdown(symptom_tags, unsafe_allow_html=True)
    
    return selected_symptoms


def display_prediction_results(predicted_disease, probabilities, selected_symptoms):
    """Display prediction results with visualizations."""
    st.markdown("## 🎯 Prediction Results")
    
    if not selected_symptoms:
        st.warning("Please select at least one symptom to get a prediction.")
        return
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <h3 style="color: #1f77b4; margin-top: 0;">Most Likely Disease</h3>
        <h2 style="color: #2c3e50; margin: 1rem 0;">{predicted_disease}</h2>
        <p style="font-size: 1.1rem;">
            <strong>Confidence:</strong> {probabilities[predicted_disease]:.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 3 predictions
    st.markdown("### 📈 Top 3 Predictions")
    top_3 = list(probabilities.items())[:3]
    
    for i, (disease, prob) in enumerate(top_3):
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.metric("Rank", f"#{i+1}")
        
        with col2:
            st.metric("Disease", disease)
        
        with col3:
            st.metric("Probability", f"{prob:.1%}")
        
        # Progress bar for probability
        st.progress(prob)
        st.markdown("---")
    
    # Interactive probability chart
    st.markdown("### 📊 Probability Distribution")
    
    # Create DataFrame for plotting
    prob_df = pd.DataFrame(list(probabilities.items()), 
                          columns=['Disease', 'Probability'])
    prob_df = prob_df.head(10)  # Show top 10
    
    # Create horizontal bar chart
    fig = px.bar(prob_df, x='Probability', y='Disease', orientation='h',
                 title="Disease Probability Distribution (Top 10)",
                 color='Probability', color_continuous_scale='Blues')
    
    fig.update_layout(
        height=400,
        xaxis_title="Probability",
        yaxis_title="Disease",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_feature_importance(model, feature_names):
    """Display feature importance analysis."""
    st.markdown("## 🔬 Model Analysis")
    
    with st.expander("Most Important Symptoms (Feature Importance)"):
        importance_df = get_feature_importance(model, feature_names, top_n=15)
        
        if importance_df is not None:
            # Create interactive chart
            fig = px.bar(importance_df.head(15), x='importance', y='feature', orientation='h',
                         title="Top 15 Most Important Symptoms",
                         color='importance', color_continuous_scale='Viridis')
            
            fig.update_layout(
                height=500,
                xaxis_title="Importance Score",
                yaxis_title="Symptom"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display as table
            st.markdown("### Feature Importance Table")
            st.dataframe(importance_df.head(15))


def display_disclaimer():
    """Display medical disclaimer."""
    st.markdown("---")
    st.markdown("""
    ## ⚠️ Important Disclaimer
    
    <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #ffc107;">
        <h4 style="color: #856404; margin-top: 0;">Medical Disclaimer</h4>
        <ul style="color: #856404;">
            <li>This tool is for <strong>educational and research purposes only</strong></li>
            <li>It should <strong>NOT be used as a substitute</strong> for professional medical advice</li>
            <li>Always consult with a qualified healthcare provider for medical concerns</li>
            <li>The predictions are based on machine learning models and may not be 100% accurate</li>
            <li>Emergency medical situations require immediate professional attention</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    # Load model components
    model, label_encoder, feature_names, performance = load_model_components()
    
    if model is None:
        st.stop()
    
    # Display header
    display_header()
    
    # Create sidebar
    display_sidebar_info(performance, feature_names)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Symptom input
        selected_symptoms = create_symptom_input(feature_names)
        
        # Prediction button
        predict_button = st.button("🔮 Predict Disease", type="primary", use_container_width=True)
        
        if predict_button and selected_symptoms:
            # Make prediction
            with st.spinner("Analyzing symptoms..."):
                predicted_disease, probabilities = predict_disease(
                    model, label_encoder, feature_names, selected_symptoms
                )
            
            # Display results
            display_prediction_results(predicted_disease, probabilities, selected_symptoms)
    
    with col2:
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        if selected_symptoms:
            st.metric("Symptoms Selected", len(selected_symptoms))
            st.metric("Total Symptoms Available", len(feature_names))
        
        # Model info
        if performance:
            st.metric("Model Accuracy", f"{performance['accuracy']:.1%}")
        
        # Recent predictions (placeholder for future enhancement)
        st.markdown("### 📝 Recent Activity")
        st.info("Prediction history feature coming soon!")
    
    # Feature importance analysis
    if selected_symptoms:
        display_feature_importance(model, feature_names)
    
    # Footer with disclaimer
    display_disclaimer()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Smart Disease Predictor | Built with ❤️ using Streamlit and Scikit-learn</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
