import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jwt
from datetime import datetime, timedelta
import os
import sys
import pickle
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import (
    cross_val_score, TimeSeriesSplit, train_test_split,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add the Django project to the path for importing models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure Streamlit page
st.set_page_config(
    page_title="MediSense Model Training",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None

def verify_token():
    """
    Simplified token verification - always allow access for now
    """
    # For development/testing, always authenticate
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = True
        st.session_state['user_data'] = {'username': 'Developer'}
    
    return True

@st.cache_data
def load_data(file_path):
    """Load and cache data from CSV files"""
    try:
        df = pd.read_csv(file_path)
        # Convert date columns to datetime
        date_cols = ['date', 'date_cleaned', 'week_start', 'week_end', 'week_start_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def add_lag_features(df, target_col='visit_count', lags=[1, 7, 14]):
    """Add lag features for time series prediction"""
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
        
    # Add rolling statistics
    for window in [7, 14, 30]:
        df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).mean()
        df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).std()
    
    # Drop rows with NaN values from lag features
    df_copy = df_copy.dropna()
    return df_copy

def prepare_features(df, target_type='regression', prediction_target='visit_count'):
    """Prepare features for model training"""
    # Identify feature columns to exclude
    exclude_cols = ['date', 'date_cleaned', 'week_start', 'week_end', 'week_start_date',
                   'normalized_symptoms', 'symptoms_combined', 'course_mapped', 
                   'courses_list', 'gender', 'genders_list', 'year_week']
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Handle different prediction targets
    if prediction_target == 'visit_count':
        # Original visit count prediction
        if target_type == 'classification':
            # Create classification target (e.g., high/low visits)
            if 'visit_count' in df.columns:
                threshold = df['visit_count'].median()
                df['target'] = (df['visit_count'] > threshold).astype(int)
                feature_cols = [col for col in feature_cols if col != 'visit_count']
        else:
            df['target'] = df['visit_count']
            feature_cols = [col for col in feature_cols if col != 'visit_count']
    
    elif prediction_target == 'symptom_category':
        # Symptom category prediction
        symptom_cols = [col for col in df.columns if any(symptom in col.lower() for symptom in 
                       ['respiratory', 'digestive', 'pain', 'fever', 'neurological', 'skin', 'cardiovascular'])]
        
        if symptom_cols:
            # Create dominant symptom category as target
            symptom_data = df[symptom_cols].fillna(0)
            df['target'] = symptom_data.idxmax(axis=1)
            
            # Convert to numeric labels for classification
            unique_symptoms = df['target'].unique()
            symptom_mapping = {symptom: idx for idx, symptom in enumerate(unique_symptoms)}
            df['target'] = df['target'].map(symptom_mapping)
            
            # Remove symptom columns from features
            feature_cols = [col for col in feature_cols if col not in symptom_cols]
        else:
            st.error("No symptom category columns found in the dataset!")
            return None, None, None
    
    elif prediction_target == 'symptom_presence':
        # Binary symptom presence prediction
        symptom_cols = [col for col in df.columns if any(symptom in col.lower() for symptom in 
                       ['respiratory', 'digestive', 'pain', 'fever', 'neurological', 'skin', 'cardiovascular'])]
        
        if symptom_cols:
            # Create binary target: any symptoms present (1) or not (0)
            symptom_data = df[symptom_cols].fillna(0)
            df['target'] = (symptom_data.sum(axis=1) > 0).astype(int)
            
            # Remove symptom columns from features
            feature_cols = [col for col in feature_cols if col not in symptom_cols]
        else:
            st.error("No symptom category columns found in the dataset!")
            return None, None, None
    
    # Remove target from features if it exists
    feature_cols = [col for col in feature_cols if col != 'target']
    
    return df[feature_cols], df['target'], feature_cols

def train_model(X_train, y_train, X_test, y_test, model_type, hyperparams, task_type='regression'):
    """Train a model with given parameters"""
    
    # Initialize model based on type and task
    if task_type == 'regression':
        if model_type == 'XGBoost':
            model = xgb.XGBRegressor(**hyperparams)
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(**hyperparams)
        elif model_type == 'Linear Regression':
            model = LinearRegression()
        else:  # Naive Bayes doesn't support regression well, use Linear as fallback
            model = LinearRegression()
    else:  # classification
        if model_type == 'XGBoost':
            model = xgb.XGBClassifier(**hyperparams)
        elif model_type == 'Random Forest':
            model = RandomForestClassifier(**hyperparams)
        elif model_type == 'Logistic Regression':
            model = LogisticRegression(**hyperparams)
        elif model_type == 'Naive Bayes':
            model = GaussianNB()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    if task_type == 'regression':
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    else:
        # For classification, also get probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
            
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
    
    return model, y_pred, metrics

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Top Feature Importances',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=400
        )
        
        return fig
    return None

def plot_predictions_vs_actual(y_test, y_pred, task_type='regression'):
    """Plot predictions vs actual values"""
    if task_type == 'regression':
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Predictions vs Actual Values',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=400
        )
    else:
        # Confusion matrix for classification
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, 
                       text_auto=True,
                       labels=dict(x="Predicted", y="Actual"),
                       title="Confusion Matrix")
    
    return fig

def plot_residuals(y_test, y_pred):
    """Plot residuals for regression models"""
    residuals = y_test - y_pred
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Residuals Distribution', 'Residuals vs Predicted'))
    
    # Histogram of residuals
    fig.add_trace(go.Histogram(x=residuals, name='Residuals', nbinsx=30),
                  row=1, col=1)
    
    # Residuals vs predicted
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                             name='Residuals', marker=dict(color='blue', size=5, opacity=0.5)),
                  row=1, col=2)
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Residual Value", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Value", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    
    return fig

def main():
    # Verify token
    if not verify_token():
        st.stop()
    
    # Get user data from session
    user_data = st.session_state.get('user_data', {'username': 'Unknown User'})
    
    # Header
    st.title("MediSense Model Training Dashboard")
    # st.markdown(f"Welcome, **{user_data.get('username', 'User')}**! Train and evaluate medical trend prediction models.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data selection
    st.sidebar.subheader("📊 Data Selection")
    data_source = st.sidebar.selectbox(
        "Select Dataset:",
        ["daily_complete.csv", "daily_visits_only.csv", 
         "weekly_complete.csv", "weekly_visits_only.csv", 
         "Upload Custom"]
    )
    
    # Prediction target selection
    st.sidebar.subheader("🎯 Prediction Target")
    prediction_target_display = st.sidebar.selectbox(
        "What to Predict:",
        ["Visit Count", "Symptom Category", "Symptom Presence"],
        help="Choose what you want to predict:\n• Visit Count: Number of clinic visits\n• Symptom Category: Which symptom type will be dominant\n• Symptom Presence: Whether any symptoms will appear"
    )
    
    # Convert display name to internal name
    if prediction_target_display == "Visit Count":
        prediction_target = 'visit_count'
    elif prediction_target_display == "Symptom Category":
        prediction_target = 'symptom_category'
    else:
        prediction_target = 'symptom_presence'
    
    # Load data
    data_loaded = False
    if data_source == "Upload Custom":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            data_loaded = True
    else:
        # Construct path to data file
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'final', data_source
        )
        if os.path.exists(data_path):
            df = load_data(data_path)
            if df is not None:
                data_loaded = True
        else:
            st.error(f"Data file not found: {data_path}")
    
    if data_loaded:
        st.session_state.current_data = df
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Data Overview", "🔧 Preprocessing", "🎯 Model Training", 
             "📈 Evaluation", "💾 Model Management"]
        )
        
        with tab1:
            st.header("Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                if 'visit_count' in df.columns:
                    st.metric("Avg Visits", f"{df['visit_count'].mean():.2f}")
            with col4:
                if 'visit_count' in df.columns:
                    st.metric("Max Visits", int(df['visit_count'].max()))
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic statistics
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Target distribution
            if 'visit_count' in df.columns:
                st.subheader("Target Variable Distribution")
                fig = px.histogram(df, x='visit_count', nbins=30,
                                 title='Visit Count Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Data Preprocessing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Engineering")
                
                # Add lag features
                add_lags = st.checkbox("Add Lag Features (for time series)", value=True)
                if add_lags:
                    lag_values = st.multiselect(
                        "Select lag periods:",
                        [1, 3, 7, 14, 21, 30],
                        default=[1, 7, 14]
                    )
                
                # Scaling
                scaling_method = st.selectbox(
                    "Feature Scaling:",
                    ["None", "StandardScaler", "MinMaxScaler"]
                )
            
            with col2:
                st.subheader("Data Splitting")
                
                # Task type - make it conditional
                if prediction_target == 'visit_count':
                    task_type = st.radio(
                        "Task Type:",
                        ["Regression", "Classification"],
                        help="Regression: Predict exact visit count\nClassification: Predict high/low visits"
                    )
                else:
                    task_type = "Classification"  # Force classification for symptoms
                    st.info(f"🎯 Task Type: **Classification** (automatic for {prediction_target_display})")
                
                # Split method
                split_method = st.selectbox(
                    "Split Method:",
                    ["Time-based Split", "Random Split"]
                )
                
                # Test size
                test_size = st.slider(
                    "Test Set Size:",
                    min_value=0.1,
                    max_value=0.4,
                    value=0.2,
                    step=0.05
                )
            
            # Preprocess button
            if st.button("Apply Preprocessing", type="primary"):
                with st.spinner("Preprocessing data..."):
                    # Apply preprocessing
                    df_processed = df.copy()
                    
                    # Add lag features if selected
                    if add_lags and 'visit_count' in df_processed.columns:
                        df_processed = add_lag_features(df_processed, 'visit_count', lag_values)
                    
                    # Determine correct task type based on prediction target
                    if prediction_target == 'visit_count':
                        actual_task_type = task_type.lower()  # User choice
                    else:
                        actual_task_type = 'classification'  # Force classification for symptoms
                    
                    # Prepare features with the correct prediction target
                    X, y, feature_names = prepare_features(
                        df_processed, 
                        target_type=actual_task_type,
                        prediction_target=prediction_target  # ✅ NOW INCLUDED!
                    )
                    
                    if X is None:  # Handle preparation errors
                        st.error("❌ Feature preparation failed!")
                        return
                    
                    # Apply scaling
                    scaler = None
                    if scaling_method != "None":
                        if scaling_method == "StandardScaler":
                            scaler = StandardScaler()
                        else:
                            scaler = MinMaxScaler()
                        X_scaled = scaler.fit_transform(X)
                        X = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
                    
                    # Store preprocessed data
                    st.session_state.preprocessed_data = {
                        'X': X,
                        'y': y,
                        'feature_names': feature_names,
                        'task_type': actual_task_type,  # Use the corrected task type
                        'scaler': scaler,
                        'split_method': split_method,
                        'test_size': test_size,
                        'prediction_target': prediction_target
                    }
                    
                    st.success(f"✅ Preprocessing complete!")
                    st.info(f"Target: {prediction_target_display} | Task: {actual_task_type} | Features: {len(feature_names)} | Samples: {len(X)}")
                    
                    # Show target distribution
                    if actual_task_type == 'classification':
                        st.subheader("Target Distribution")
                        target_counts = pd.Series(y).value_counts()
                        st.write(target_counts)
                        
                        # Check for class imbalance
                        if len(target_counts) > 1:
                            imbalance_ratio = target_counts.max() / target_counts.min()
                            if imbalance_ratio > 5:
                                st.warning(f"⚠️ Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
        
                        # Show feature list
                        with st.expander("View Selected Features"):
                            st.write(feature_names)
        
        with tab3:
            st.header("Model Training")
            
            if st.session_state.preprocessed_data is None:
                st.warning("⚠️ Please preprocess the data first in the Preprocessing tab.")
            else:
                data = st.session_state.preprocessed_data
                X = data['X']
                y = data['y']
                feature_names = data['feature_names']
                task_type = data['task_type']
                split_method = data['split_method']
                test_size = data['test_size']
                prediction_target = data['prediction_target']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Model Selection")
                    
                    # Model type selection
                    if task_type == 'regression':
                        model_options = ['XGBoost', 'Random Forest', 'Linear Regression']
                    else:
                        model_options = ['XGBoost', 'Random Forest', 'Logistic Regression', 'Naive Bayes']
                    
                    model_type = st.selectbox("Select Model:", model_options)
                    
                    # Hyperparameters
                    st.subheader("Hyperparameters")
                    hyperparams = {}
                    
                    if model_type == 'XGBoost':
                        hyperparams['n_estimators'] = st.slider("Number of Trees:", 50, 500, 100)
                        hyperparams['max_depth'] = st.slider("Max Depth:", 3, 10, 5)
                        hyperparams['learning_rate'] = st.slider("Learning Rate:", 0.01, 0.3, 0.1)
                        hyperparams['subsample'] = st.slider("Subsample:", 0.5, 1.0, 0.8)
                    elif model_type == 'Random Forest':
                        hyperparams['n_estimators'] = st.slider("Number of Trees:", 50, 500, 100)
                        hyperparams['max_depth'] = st.slider("Max Depth:", 3, 20, 10)
                        hyperparams['min_samples_split'] = st.slider("Min Samples Split:", 2, 10, 2)
                        hyperparams['min_samples_leaf'] = st.slider("Min Samples Leaf:", 1, 10, 1)
                    elif model_type == 'Logistic Regression':
                        hyperparams['C'] = st.slider("Regularization (C):", 0.01, 10.0, 1.0)
                        hyperparams['max_iter'] = st.slider("Max Iterations:", 100, 1000, 100)
                
                with col2:
                    st.subheader("Training Configuration")
                    
                    # Cross-validation
                    use_cv = st.checkbox("Use Cross-Validation", value=True)
                    if use_cv:
                        cv_folds = st.slider("Number of Folds:", 3, 10, 5)
                    
                    # Class balancing (for classification)
                    if task_type == 'classification':
                        balance_classes = st.checkbox("Balance Classes", value=True)
                        if balance_classes and model_type in ['Random Forest', 'Logistic Regression']:
                            hyperparams['class_weight'] = 'balanced'
                
                # Train button
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner(f"Training {model_type} model..."):
                        # Split data
                        if split_method == "Time-based Split":
                            # Assuming data is sorted by time
                            split_idx = int(len(X) * (1 - test_size))
                            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42
                            )
                        
                        # Train model
                        model, y_pred, metrics = train_model(
                            X_train, y_train, X_test, y_test,
                            model_type, hyperparams, task_type
                        )
                        
                        # Store model in session state
                        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.trained_models[model_id] = {
                            'model': model,
                            'type': model_type,
                            'metrics': metrics,
                            'feature_names': feature_names,
                            'task_type': task_type,
                            'hyperparams': hyperparams,
                            'timestamp': datetime.now(),
                            'prediction_target': prediction_target
                        }
                        
                        # Store test data for evaluation tab
                        st.session_state.last_training = {
                            'model': model,
                            'X_test': X_test,
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'feature_names': feature_names,
                            'task_type': task_type,
                            'prediction_target': prediction_target
                        }
                        
                        st.success(f"✅ Model trained successfully! Model ID: {model_id}")
                        
                        # Display metrics
                        st.subheader("Training Results")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        if task_type == 'regression':
                            with col1:
                                st.metric("RMSE", f"{metrics['rmse']:.4f}")
                            with col2:
                                st.metric("MAE", f"{metrics['mae']:.4f}")
                            with col3:
                                st.metric("R² Score", f"{metrics['r2']:.4f}")
                            with col4:
                                st.metric("MSE", f"{metrics['mse']:.4f}")
                        else:
                            with col1:
                                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            with col2:
                                st.metric("Precision", f"{metrics['precision']:.4f}")
                            with col3:
                                st.metric("Recall", f"{metrics['recall']:.4f}")
                            with col4:
                                st.metric("F1 Score", f"{metrics['f1']:.4f}")
        
        with tab4:
            st.header("Model Evaluation")
            
            if 'last_training' not in st.session_state:
                st.warning("⚠️ Please train a model first in the Model Training tab.")
            else:
                training_data = st.session_state.last_training
                model = training_data['model']
                X_test = training_data['X_test']
                y_test = training_data['y_test']
                y_pred = training_data['y_pred']
                feature_names = training_data['feature_names']
                task_type = training_data['task_type']
                prediction_target = training_data['prediction_target']
                
                # Predictions vs Actual
                st.subheader("Predictions vs Actual")
                fig = plot_predictions_vs_actual(y_test, y_pred, task_type)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    fig = plot_feature_importance(model, feature_names)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Residuals (for regression)
                if task_type == 'regression':
                    st.subheader("Residual Analysis")
                    fig = plot_residuals(y_test, y_pred)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model comparison
                if len(st.session_state.trained_models) > 1:
                    st.subheader("Model Comparison")
                    
                    comparison_data = []
                    for model_id, model_info in st.session_state.trained_models.items():
                        row = {'Model ID': model_id, 'Type': model_info['type']}
                        row.update(model_info['metrics'])
                        comparison_data.append(row)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
        
        with tab5:
            st.header("Model Management")
            
            if len(st.session_state.trained_models) == 0:
                st.info("No trained models yet.")
            else:
                for model_id, model_info in st.session_state.trained_models.items():
                    with st.expander(f"{model_id}"):
                        st.write(f"**Type:** {model_info['type']}")
                        st.write(f"**Task:** {model_info['task_type']}")
                        st.write(f"**Prediction Target:** {model_info['prediction_target']}")
                        st.write(f"**Trained:** {model_info['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.write("**Metrics:**")
                        for metric, value in model_info['metrics'].items():
                            st.write(f"  - {metric}: {value:.4f}")
                        
                        # Save model button
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Save {model_id}", key=f"save_{model_id}"):
                                # Create absolute path for models directory
                                models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                                os.makedirs(models_dir, exist_ok=True)
                                
                                model_path = os.path.join(models_dir, f"{model_id}.pkl")
                                
                                try:
                                    # Save the entire model info, not just the model
                                    save_data = {
                                        'model': model_info['model'],
                                        'type': model_info['type'], 
                                        'metrics': model_info['metrics'],
                                        'feature_names': model_info['feature_names'],
                                        'task_type': model_info['task_type'],
                                        'hyperparams': model_info['hyperparams'],
                                        'timestamp': model_info['timestamp'],
                                        'prediction_target': model_info['prediction_target']
                                    }
                                    
                                    joblib.dump(save_data, model_path)
                                    st.success(f"✅ Model saved successfully!")
                                    st.info(f"📁 Location: {model_path}")
                                    
                                    # Also show in expander for easy copy
                                    with st.expander("📋 Full Path (click to copy)"):
                                        st.code(model_path)
                                        
                                except Exception as e:
                                    st.error(f"❌ Error saving model: {str(e)}")
                                    st.error(f"Attempted to save to: {model_path}")
                        
                        with col2:
                            if st.button(f"Delete {model_id}", key=f"delete_{model_id}"):
                                del st.session_state.trained_models[model_id]
                                st.rerun()
            
            st.subheader("📂 Load Saved Models")
            
            # Get models directory
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            
            if os.path.exists(models_dir):
                saved_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                
                if saved_models:
                    st.write(f"Found {len(saved_models)} saved models in: `{models_dir}`")
                    
                    selected_model = st.selectbox("Select model to load:", saved_models)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Load Model"):
                            try:
                                model_path = os.path.join(models_dir, selected_model)
                                loaded_data = joblib.load(model_path)
                                
                                # Add to session state
                                model_id = selected_model.replace('.pkl', '')
                                st.session_state.trained_models[model_id] = loaded_data
                                
                                st.success(f"✅ Model loaded: {model_id}")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error loading model: {str(e)}")
                    
                    with col2:
                        if st.button("🗑️ Delete Saved Model"):
                            try:
                                model_path = os.path.join(models_dir, selected_model)
                                os.remove(model_path)
                                st.success(f"✅ Deleted: {selected_model}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting: {str(e)}")
                else:
                    st.info("No saved models found.")
            else:
                st.info(f"Models directory doesn't exist yet: `{models_dir}`")
                st.write("Train and save a model first to create the directory.")
    else:
        st.error("Please select or upload a dataset to begin.")

if __name__ == "__main__":
    main()
