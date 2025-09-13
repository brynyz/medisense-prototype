import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jwt
from datetime import datetime
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Add the Django project to the path for importing models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure Streamlit page
st.set_page_config(
    page_title="MediSense Model Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Import Line Awesome and Material Icons to match Django app */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/line-awesome/1.3.0/line-awesome/css/line-awesome.min.css');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    @import url("https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap");
    
    /* Match Django app variables */
    :root {
        --color-main: #0a400c;
        --transition-speed: 300ms;
        --transition-easing: ease-in-out;
        --transition-all: all var(--transition-speed) var(--transition-easing);
    }
    
    /* Override Streamlit default fonts */
    .main, .sidebar .sidebar-content {
        font-family: "Poppins", sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .main-header h1::before {
        content: "";
        font-family: "Line Awesome Free";
        font-weight: 900;
        font-size: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: var(--transition-all);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: var(--transition-all);
    }
    
    .stMetric:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Icon styles */
    .icon {
        font-family: "Line Awesome Free";
        font-weight: 900;
        margin-right: 0.5rem;
    }
    
    .icon-target::before { content: "\\f140"; }
    .icon-search::before { content: "\\f002"; }
    .icon-chart::before { content: "\\f080"; }
    .icon-balance::before { content: "\\f24e"; }
    .icon-analytics::before { content: "\\f201"; }
    .icon-microscope::before { content: "\\f610"; }
    .icon-brain::before { content: "\\f5dc"; }
    .icon-lock::before { content: "\\f023"; }
    .icon-cog::before { content: "\\f013"; }
    .icon-refresh::before { content: "\\f021"; }
    .icon-info::before { content: "\\f129"; }
    .icon-wave::before { content: "\\f1fe"; }
    
    /* Alert styles matching Django */
    .stAlert {
        border-radius: 8px;
        border: none;
        font-family: "Poppins", sans-serif;
    }
    
    .stAlert[data-baseweb="notification"] {
        background-color: #d1ecf1;
        color: #0c5460;
        border-left: 4px solid #bee5eb;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-family: "Poppins", sans-serif;
        transition: var(--transition-all);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def verify_token():
    """Verify JWT token from Django"""
    # Development mode bypass
    if os.getenv('STREAMLIT_DEV_MODE', 'false').lower() == 'true':
        st.info("🔧 Running in development mode - authentication bypassed")
        return {
            'user_id': 1,
            'username': 'dev_user',
            'email': 'dev@example.com',
            'is_superuser': True
        }
    
    query_params = st.query_params
    token = query_params.get('token', None)
    
    if not token:
        st.error("🔒 Authentication required. Please access this page through the main application.")
        st.info("💡 For development, set STREAMLIT_DEV_MODE=true environment variable")
        st.stop()
    
    try:
        # Get Django secret key from environment or use fallback
        SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')
        if not SECRET_KEY:
            # Try to import Django settings directly
            try:
                import django
                from django.conf import settings
                os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medisense.settings')
                django.setup()
                SECRET_KEY = settings.SECRET_KEY
            except Exception as e:
                st.error(f"🔒 Configuration error: Unable to load Django secret key. {str(e)}")
                st.stop()
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        st.error("🔒 Session expired. Please refresh the page.")
        st.stop()
    except jwt.InvalidTokenError as e:
        st.error(f"🔒 Invalid authentication token. Error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"🔒 Authentication error: {str(e)}")
        st.stop()

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Sample model performance data
    models = ['XGBoost', 'Random Forest', 'Logistic Regression', 'Naive Bayes']
    metrics = {
        'Model': models,
        'Accuracy': [0.87, 0.84, 0.79, 0.75],
        'Precision': [0.86, 0.83, 0.78, 0.74],
        'Recall': [0.88, 0.85, 0.80, 0.76],
        'F1-Score': [0.87, 0.84, 0.79, 0.75],
        'AUC-ROC': [0.92, 0.89, 0.84, 0.80]
    }
    
    return pd.DataFrame(metrics)

def create_confusion_matrix_plot(model_name):
    """Create confusion matrix visualization"""
    # Sample confusion matrix data
    np.random.seed(42)
    cm = np.array([[85, 12], [8, 95]])
    
    fig = px.imshow(cm, 
                    text_auto=True, 
                    aspect="auto",
                    color_continuous_scale='Blues',
                    title=f'Confusion Matrix - {model_name}')
    
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis={'tickmode': 'array', 'tickvals': [0, 1], 'ticktext': ['No Risk', 'Risk']},
        yaxis={'tickmode': 'array', 'tickvals': [0, 1], 'ticktext': ['No Risk', 'Risk']}
    )
    
    return fig

def create_roc_curve_plot():
    """Create ROC curve comparison"""
    fig = go.Figure()
    
    # Sample ROC data for different models
    models_roc = {
        'XGBoost': {'fpr': [0, 0.1, 0.2, 0.4, 1], 'tpr': [0, 0.8, 0.9, 0.95, 1], 'auc': 0.92},
        'Random Forest': {'fpr': [0, 0.15, 0.25, 0.45, 1], 'tpr': [0, 0.75, 0.85, 0.92, 1], 'auc': 0.89},
        'Logistic Regression': {'fpr': [0, 0.2, 0.35, 0.5, 1], 'tpr': [0, 0.7, 0.8, 0.88, 1], 'auc': 0.84},
        'Naive Bayes': {'fpr': [0, 0.25, 0.4, 0.6, 1], 'tpr': [0, 0.65, 0.75, 0.85, 1], 'auc': 0.80}
    }
    
    colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b']
    
    for i, (model, data) in enumerate(models_roc.items()):
        fig.add_trace(go.Scatter(
            x=data['fpr'], 
            y=data['tpr'],
            mode='lines',
            name=f"{model} (AUC = {data['auc']:.2f})",
            line=dict(color=colors[i], width=3)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        width=600,
        height=500
    )
    
    return fig

def create_feature_importance_plot():
    """Create feature importance visualization"""
    features = ['Age', 'BMI', 'Blood Pressure', 'Heart Rate', 'Temperature', 
                'Symptoms Count', 'Visit Frequency', 'Medication History']
    importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    
    fig = px.bar(
        x=importance, 
        y=features,
        orientation='h',
        title='Feature Importance (XGBoost)',
        color=importance,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Features',
        showlegend=False
    )
    
    return fig

def create_model_comparison_radar():
    """Create radar chart for model comparison"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    models_data = {
        'XGBoost': [0.87, 0.86, 0.88, 0.87, 0.92],
        'Random Forest': [0.84, 0.83, 0.85, 0.84, 0.89],
        'Logistic Regression': [0.79, 0.78, 0.80, 0.79, 0.84],
        'Naive Bayes': [0.75, 0.74, 0.76, 0.75, 0.80]
    }
    
    fig = go.Figure()
    
    colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b']
    
    for i, (model, values) in enumerate(models_data.items()):
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=model,
            line_color=colors[i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Comparison"
    )
    
    return fig

def main():
    # Verify authentication
    user_data = verify_token()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1><i class="las la-hospital"></i>MediSense Model Analytics Dashboard</h1>
        <p>Comprehensive analysis of medical trend prediction models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.info(f"👋 Welcome back, {user_data.get('username', 'User')}!")
    
    # Sidebar for model selection and filters
    st.sidebar.header("🔧 Dashboard Controls")
    
    selected_model = st.sidebar.selectbox(
        "Select Model for Detailed Analysis:",
        ['XGBoost', 'Random Forest', 'Logistic Regression', 'Naive Bayes']
    )
    
    time_period = st.sidebar.selectbox(
        "Time Period:",
        ['Last 30 days', 'Last 90 days', 'Last 6 months', 'Last year']
    )
    
    show_advanced = st.sidebar.checkbox("Show Advanced Metrics", value=True)
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate sample data
    df_metrics = generate_sample_data()
    selected_metrics = df_metrics[df_metrics['Model'] == selected_model].iloc[0]
    
    with col1:
        st.metric(
            label="🎯 Accuracy", 
            value=f"{selected_metrics['Accuracy']:.1%}",
            delta="2.3%"
        )
    
    with col2:
        st.metric(
            label="🔍 Precision", 
            value=f"{selected_metrics['Precision']:.1%}",
            delta="1.8%"
        )
    
    with col3:
        st.metric(
            label="📊 Recall", 
            value=f"{selected_metrics['Recall']:.1%}",
            delta="3.1%"
        )
    
    with col4:
        st.metric(
            label="⚖️ F1-Score", 
            value=f"{selected_metrics['F1-Score']:.1%}",
            delta="2.7%"
        )
    
    st.markdown("---")
    
    # Model comparison section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Performance Comparison")
        st.plotly_chart(create_model_comparison_radar(), use_container_width=True)
    
    with col2:
        st.subheader("📊 Performance Metrics Table")
        st.dataframe(df_metrics.set_index('Model'), use_container_width=True)
    
    # ROC Curve and Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 ROC Curve Analysis")
        st.plotly_chart(create_roc_curve_plot(), use_container_width=True)
    
    with col2:
        st.subheader(f"🔍 Confusion Matrix - {selected_model}")
        st.plotly_chart(create_confusion_matrix_plot(selected_model), use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance Analysis")
    st.plotly_chart(create_feature_importance_plot(), use_container_width=True)
    
    if show_advanced:
        st.markdown("---")
        st.subheader("🔬 Advanced Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cross-Validation Score", "0.85 ± 0.03")
            st.metric("Training Time", "2.3 seconds")
        
        with col2:
            st.metric("Prediction Time", "0.05 ms")
            st.metric("Model Size", "1.2 MB")
        
        with col3:
            st.metric("Data Points", "10,547")
            st.metric("Features", "8")
        
        # Model interpretation
        st.subheader("🧠 Model Interpretation")
        st.info("""
        **Key Insights:**
        - Age and BMI are the strongest predictors for medical risk assessment
        - The XGBoost model shows superior performance across all metrics
        - Feature interactions reveal complex patterns in patient data
        - Model confidence is highest for extreme risk cases
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>MediSense Analytics Dashboard | Last updated: {} | User: {}</small>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M"), user_data.get('username')), 
    unsafe_allow_html=True)

if __name__ == "__main__":
    main()
