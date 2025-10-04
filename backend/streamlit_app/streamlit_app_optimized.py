"""
MEDISENSE OPTIMIZED PREDICTION SYSTEM
Two-Stage Medical Trends Prediction with Streamlit
Leveraging lag features, environmental interactions, and balanced datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MediSense Analytics - Optimized",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Custom CSS for better UI
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1e3d59;
#         text-align: center;
#         padding: 1rem 0;
#         border-bottom: 3px solid #ff6e40;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f5f5f5;
#         padding: 1.5rem;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         margin-bottom: 1rem;
#     }
#     .stage-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .insight-box {
#         background-color: #e8f4f8;
#         border-left: 4px solid #3498db;
#         padding: 1rem;
#         margin: 1rem 0;
#         border-radius: 5px;
#     }
#     .warning-box {
#         background-color: #fff3cd;
#         border-left: 4px solid #ffc107;
#         padding: 1rem;
#         margin: 1rem 0;
#         border-radius: 5px;
#     }
#     .success-box {
#         background-color: #d4edda;
#         border-left: 4px solid #28a745;
#         padding: 1rem;
#         margin: 1rem 0;
#         border-radius: 5px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'stage1_model' not in st.session_state:
    st.session_state.stage1_model = None
if 'stage2_models' not in st.session_state:
    st.session_state.stage2_models = {}

# Title
st.markdown('<h1 class="main-header">🏥 MediSense Optimized Prediction System</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# Model selection
st.sidebar.markdown("### 🎯 Two-Stage Prediction Strategy")
st.sidebar.info("""
**Stage 1:** Visit Prediction
- Binary classification (visit/no-visit)
- Regression (visit volume)

**Stage 2:** Symptom Classification
- Only for predicted visit days
- Respiratory detection
- Dominant symptom prediction
- Outbreak detection
""")

# Load optimized datasets
@st.cache_data
def load_datasets():
    """Load all optimized datasets"""
    datasets = {}
    
    # Stage 1 datasets
    stage1_files = {
        'binary_visits': 'dataset_binary_visits.csv',
        'regression_visits': 'dataset_regression_visits.csv',
        'multiclass_visits': 'dataset_multiclass_visits.csv',
        'risk_based': 'dataset_risk_based.csv'
    }
    
    # Stage 2 datasets
    stage2_files = {
        'respiratory_detection': 'dataset_binary_respiratory_present.csv',
        'dominant_symptom': 'dataset_dominant_symptom.csv',
        'respiratory_outbreak': 'dataset_respiratory_outbreak.csv',
        'symptom_severity': 'dataset_symptom_severity.csv'
    }
    
    # Fix the base path - adjust based on where you're running from
    import os
    current_dir = os.getcwd()
    
    # Try different possible paths
    possible_paths = [
        'data/final/',  # If running from medisense/backend/
        'medisense/backend/data/final/',  # If running from project root
        '../data/final/',  # If running from streamlit_app folder
        './data/final/',  # Current directory
    ]
    
    base_path = None
    for path in possible_paths:
        test_file = os.path.join(path, 'dataset_binary_visits.csv')
        if os.path.exists(test_file):
            base_path = path
            break
    
    if base_path is None:
        st.error("❌ Could not find data/final/ directory. Please check file paths.")
        st.info(f"Current working directory: {current_dir}")
        return {}
    
    st.sidebar.success(f"📂 Found data directory: {base_path}")
    
    # Load all datasets
    for name, file in {**stage1_files, **stage2_files}.items():
        try:
            filepath = os.path.join(base_path, file)
            df = pd.read_csv(filepath)
            
            # Convert date column if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            datasets[name] = df
            st.sidebar.success(f"✅ Loaded {name} ({len(df)} rows)")
            
        except Exception as e:
            st.sidebar.error(f"❌ Could not load {name}: {str(e)}")
    
    return datasets
# Main app tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "🎯 Stage 1: Visit Prediction",
    "🔬 Stage 2: Symptom Classification", 
    "📈 Model Performance",
    "🔮 Real-time Predictions"
])

# Load data
with st.spinner("Loading optimized datasets..."):
    datasets = load_datasets()

# Tab 1: Data Overview
with tab1:
    st.header("📊 Data Overview & Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Key Insights from Optimization")
        st.markdown("""
        - **Two-stage strategy** is excellent for medical prediction
        - **Symptom classification** focused only on visit days (better balance)
        - **Environmental interactions** improve symptom prediction
        - **Lag features** capture important temporal patterns
        - **Multiple target formulations** for different use cases
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Data Challenges Addressed")
        st.markdown("""
        - **Class imbalance:** 60-70% zero-visit days → Balanced datasets
        - **Missing lag features:** → Added 1, 3, 7, 14-day lags
        - **Poor targets:** → Multiple optimized target variables
        - **Temporal leakage:** → Time series validation
        - **Missing values:** → Smart domain-specific imputation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset statistics
    st.markdown("### 📈 Dataset Statistics")
    
    if 'binary_visits' in datasets:
        df = datasets['binary_visits']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Days", len(df))
        with col2:
            visit_rate = (df['target'].mean() * 100) if 'target' in df.columns else 0
            st.metric("Visit Rate", f"{visit_rate:.1f}%")
        with col3:
            feature_count = len([col for col in df.columns if col not in ['date', 'target']])
            st.metric("Features", feature_count)
        with col4:
            st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        
        # Feature importance preview
        st.markdown("### 🔍 Feature Categories")
        
        feature_categories = {
            'Temporal': ['day_of_week', 'month', 'is_weekend', 'academic_period'],
            'Lag Features': [col for col in df.columns if 'lag' in col],
            'Rolling Features': [col for col in df.columns if 'rolling' in col],
            'Environmental': [col for col in df.columns if any(x in col for x in ['temp', 'humid', 'pm', 'aqi'])],
            'Symptom Categories': ['respiratory_count', 'digestive_count', 'pain_count', 'fever_count'],
            'Interactions': [col for col in df.columns if '_x_' in col or 'interaction' in col]
        }
        
        cols = st.columns(3)
        for idx, (category, features) in enumerate(feature_categories.items()):
            with cols[idx % 3]:
                st.markdown(f"**{category}:** {len([f for f in features if f in df.columns])} features")

# Tab 2: Stage 1 - Visit Prediction
with tab2:
    st.markdown('<div class="stage-header"><h2>🎯 Stage 1: Visit Prediction</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Model Configuration")
        
        # Visit prediction uses regression only
        st.markdown("### Visit Count Prediction")
        model_type = st.selectbox(
            "Select Model",
            ["XGBoost Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"]
        )
        dataset_key = 'regression_visits'
        target_col = 'target'
        prediction_type = "Regression"
        
        # Training parameters
        st.markdown("### Training Parameters")
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2)
        use_time_series_split = st.checkbox("Use Time Series Split", value=True)
        n_splits = st.slider("Number of CV Folds", 3, 10, 5)
        
        # Feature selection
        st.markdown("### Feature Selection")
        if dataset_key in datasets:
            df_temp = datasets[dataset_key]
            available_features = [col for col in df_temp.columns if col not in ['date', 'target']]
            
            # Group features by category for easier selection
            feature_groups = {
                'Temporal': [f for f in available_features if any(x in f for x in ['day', 'week', 'month', 'year', 'academic', 'exam', 'break', 'sin_', 'cos_'])],
                'Environmental': [f for f in available_features if any(x in f for x in ['temp', 'humid', 'pm', 'precipitation', 'wind', 'heat', 'pollution', 'weather', 'rainy', 'hot'])],
                'Lag Features': [f for f in available_features if 'lag' in f],
                'Rolling Features': [f for f in available_features if 'rolling' in f],
                'Symptom Features': [f for f in available_features if any(x in f for x in ['respiratory', 'digestive', 'pain', 'fever', 'symptom', 'diversity', 'dominance'])],
                'Interaction Features': [f for f in available_features if 'interaction' in f or 'combo' in f or 'risk' in f]
            }
            
            # Quick presets
            preset = st.selectbox(
                "Quick Presets",
                ["Use All Features", "Basic (No Interactions)", "Temporal Only", "Environmental Focus", "Lag Features Focus", "Custom"]
            )
            
            if preset == "Basic (No Interactions)":
                excluded_groups = ['Interaction Features']
            elif preset == "Temporal Only":
                excluded_groups = ['Environmental', 'Symptom Features', 'Interaction Features']
            elif preset == "Environmental Focus":
                excluded_groups = ['Lag Features', 'Rolling Features']
            elif preset == "Lag Features Focus":
                excluded_groups = ['Environmental', 'Interaction Features']
            elif preset == "Custom":
                # Option to exclude entire feature groups
                st.write("**Exclude Feature Groups:**")
                excluded_groups = st.multiselect(
                    "Select feature groups to exclude",
                    options=list(feature_groups.keys()),
                    default=[]
                )
            else:  # Use All Features
                excluded_groups = []
            
            # Get features to exclude based on selected groups
            features_to_exclude = []
            for group in excluded_groups:
                features_to_exclude.extend(feature_groups[group])
            
            # Option to exclude specific features
            remaining_features = [f for f in available_features if f not in features_to_exclude]
            specific_excludes = st.multiselect(
                "Or exclude specific features",
                options=remaining_features,
                default=[]
            )
            features_to_exclude.extend(specific_excludes)
            
            # Show summary
            selected_features = [f for f in available_features if f not in features_to_exclude]
            st.info(f"Using {len(selected_features)} out of {len(available_features)} features")
    
    with col2:
        st.markdown("### Model Training & Evaluation")
        
        if st.button("🚀 Train Stage 1 Model", type="primary"):
            if dataset_key in datasets:
                df = datasets[dataset_key]
                
                # Prepare features and target
                # Use selected features if feature selection was done
                if 'selected_features' in locals():
                    feature_cols = selected_features
                else:
                    feature_cols = [col for col in df.columns if col not in ['date', target_col]]
                
                X = df[feature_cols]
                y = df[target_col] if target_col in df.columns else None
                
                st.write(f"Training with {len(feature_cols)} features")
                
                if y is not None:
                    # Handle categorical targets
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        st.info(f"Encoded categories: {dict(zip(le.classes_, range(len(le.classes_))))}")
                    with st.spinner(f"Training {model_type}..."):
                        # Initialize model
                        if model_type == "XGBoost":
                            model = xgb.XGBClassifier(random_state=42, n_estimators=100)
                        elif model_type == "Random Forest":
                            model = RandomForestClassifier(random_state=42, n_estimators=100)
                        elif model_type == "Logistic Regression":
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            model = LogisticRegression(random_state=42, max_iter=1000)
                            X = X_scaled
                        elif model_type == "Gradient Boosting":
                            model = GradientBoostingClassifier(random_state=42, n_estimators=100)
                        elif "Regressor" in model_type:
                            if "XGBoost" in model_type:
                                model = xgb.XGBRegressor(random_state=42, n_estimators=100)
                            elif "Random Forest" in model_type:
                                model = RandomForestRegressor(random_state=42, n_estimators=100)
                            else:
                                from sklearn.ensemble import GradientBoostingRegressor
                                model = GradientBoostingRegressor(random_state=42, n_estimators=100)
                        
                        # Time series split
                        if use_time_series_split:
                            tscv = TimeSeriesSplit(n_splits=n_splits)
                            scores = cross_val_score(model, X, y, cv=tscv, 
                                                   scoring='accuracy' if 'Classification' in prediction_type else 'r2')
                            
                            # Train on full data
                            model.fit(X, y)
                            st.session_state.stage1_model = model
                            st.session_state.models_trained = True
                            
                            # Display results
                            st.success("Model trained successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean R² Score", f"{scores.mean():.3f}")
                            with col2:
                                st.metric("Std Dev", f"{scores.std():.3f}")
                            with col3:
                                st.metric("Best R² Score", f"{scores.max():.3f}")
                            
                            # Feature importance
                            if hasattr(model, 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    'feature': feature_cols,
                                    'importance': model.feature_importances_
                                }).sort_values('importance', ascending=False).head(15)
                                
                                fig = px.bar(importance_df, x='importance', y='feature', 
                                           orientation='h', title="Top 15 Feature Importances")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Please implement standard train-test split")
                else:
                    st.error(f"Target column '{target_col}' not found in dataset")
            else:
                st.error(f"Dataset '{dataset_key}' not loaded")

# Tab 3: Stage 2 - Symptom Classification
with tab3:
    st.markdown('<div class="stage-header"><h2>🔬 Stage 2: Symptom Classification (Visit Days Only)</h2></div>', 
                unsafe_allow_html=True)
    
    st.info("""
    💡 **Key Innovation:** Stage 2 models are trained only on days with visits, 
    dramatically improving class balance and prediction accuracy for symptom patterns.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Classification Tasks")
        
        task = st.selectbox(
            "Select Task",
            [
                "Respiratory Detection",
                "Dominant Symptom Category",
                "Respiratory Outbreak Detection",
                "Symptom Severity Classification"
            ]
        )
        
        # Map task to dataset
        task_dataset_map = {
            "Respiratory Detection": "respiratory_detection",
            "Dominant Symptom Category": "dominant_symptom",
            "Respiratory Outbreak Detection": "respiratory_outbreak",
            "Symptom Severity Classification": "symptom_severity"
        }
        
        dataset_key = task_dataset_map[task]
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["XGBoost", "Random Forest", "Gradient Boosting", "Naive Bayes"]
        )
        
        st.markdown("### Expected Performance")
        performance_ranges = {
            "Respiratory Detection": "70-80%",
            "Dominant Symptom Category": "65-75%",
            "Respiratory Outbreak Detection": "75-85%",
            "Symptom Severity Classification": "70-80%"
        }
        
        st.success(f"Expected Accuracy: {performance_ranges[task]}")
    
    with col2:
        st.markdown("### Model Training & Evaluation")
        
        if st.button("🚀 Train Stage 2 Model", type="primary"):
            if dataset_key in datasets:
                df = datasets[dataset_key]
                
                # Determine target column based on task
                if "respiratory" in dataset_key.lower():
                    target_col = 'has_respiratory' if 'has_respiratory' in df.columns else 'target'
                elif "dominant" in dataset_key.lower():
                    target_col = 'target'
                elif "severity" in dataset_key.lower():
                    target_col = 'target'
                else:
                    target_col = list(df.columns)[-1]  # Assume last column is target
                
                if target_col in df.columns:
                    # Prepare features and target
                    feature_cols = [col for col in df.columns if col not in ['date', target_col]]
                    X = df[feature_cols]
                    y = df[target_col]

                    # Handle categorical targets  
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        st.info(f"Encoded categories: {dict(zip(le.classes_, range(len(le.classes_))))}")
                    
                    # Check for data issues
                    st.write(f"Target shape: {y.shape}, Unique values: {len(np.unique(y))}")
                    st.write(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
                    
                    # Remove any NaN values
                    mask = ~pd.isna(y)
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    if len(y_clean) == 0:
                        st.error("No valid target values after removing NaN")
                    else:
                        with st.spinner(f"Training {model_type} for {task}..."):
                            # Initialize model
                            if model_type == "XGBoost":
                                # For multiclass, ensure we specify the number of classes
                                n_classes = len(np.unique(y_clean))
                                model = xgb.XGBClassifier(
                                    random_state=42, 
                                    n_estimators=100,
                                    objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
                                    eval_metric='mlogloss' if n_classes > 2 else 'logloss'
                                )
                            elif model_type == "Random Forest":
                                model = RandomForestClassifier(random_state=42, n_estimators=100)
                            elif model_type == "Gradient Boosting":
                                model = GradientBoostingClassifier(random_state=42, n_estimators=100)
                            elif model_type == "Naive Bayes":
                                model = GaussianNB()
                            
                            # Time series cross-validation with error handling
                            try:
                                tscv = TimeSeriesSplit(n_splits=min(5, len(np.unique(y_clean))))
                                scores = cross_val_score(model, X_clean, y_clean, cv=tscv, scoring='accuracy')
                            except Exception as e:
                                st.error(f"Cross-validation failed: {str(e)}")
                                # Try simple train-test split instead
                                from sklearn.model_selection import train_test_split
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
                                )
                                model.fit(X_train, y_train)
                                score = model.score(X_test, y_test)
                                scores = np.array([score])
                            
                            # Train on full data
                            model.fit(X_clean, y_clean)
                            st.session_state.stage2_models[task] = model
                            
                            # Display results
                            st.success(f"✅ {task} model trained successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Accuracy", f"{scores.mean():.3f}")
                            with col2:
                                st.metric("Std Dev", f"{scores.std():.3f}")
                            with col3:
                                st.metric("Best Score", f"{scores.max():.3f}")
                            
                            # Class distribution
                            if hasattr(pd.Series(y_clean), 'value_counts'):
                                st.markdown("### Class Distribution")
                                class_dist = pd.Series(y_clean).value_counts()
                                fig = px.pie(values=class_dist.values, names=class_dist.index,
                                           title=f"{task} Class Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Target column '{target_col}' not found")
            else:
                st.error(f"Dataset '{dataset_key}' not loaded")

# Tab 4: Model Performance
with tab4:
    # Import and use the performance tab module
    try:
        from model_performance_tab import render_performance_tab
        render_performance_tab(datasets, st.session_state)
    except ImportError:
        # Fallback to comprehensive inline metrics
        st.header("📈 Comprehensive Model Performance")
        
        if st.session_state.models_trained:
            # Create sub-tabs for different analyses
            perf_tab1, perf_tab2 = st.tabs([
                "Regression Metrics", 
                "Visualizations"
            ])
            
            with perf_tab1:
                st.markdown("### Visit Count Regression Performance")
                
                if 'regression_visits' in datasets and hasattr(st.session_state, 'stage1_model'):
                    df = datasets['regression_visits']
                    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
                    X = df[feature_cols]
                    y = df['target']
                    
                    # Handle both dictionary and direct model storage
                    if isinstance(st.session_state.stage1_model, dict):
                        model = st.session_state.stage1_model.get('model')
                    else:
                        model = st.session_state.stage1_model
                    if model:
                        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
                        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                        
                        # Use same evaluation as Stage 1 - TimeSeriesSplit
                        tscv = TimeSeriesSplit(n_splits=5)
                        scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                        
                        # Display metrics matching Stage 1
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean R² Score", f"{scores.mean():.3f}")
                        with col2:
                            st.metric("Std Dev", f"{scores.std():.3f}")
                        with col3:
                            st.metric("Best R² Score", f"{scores.max():.3f}")
                        
                else:
                    st.error("No regression dataset available or model not trained.")
            
            with perf_tab2:
                st.markdown("### Visualizations")
                
                viz_option = st.selectbox("Select Visualization", 
                    ["Feature Importance", "Predicted vs Actual"])
                
                if viz_option == "Feature Importance" and hasattr(st.session_state, 'stage1_model'):
                    # Handle both dictionary and direct model storage
                    if isinstance(st.session_state.stage1_model, dict):
                        model = st.session_state.stage1_model.get('model')
                    else:
                        model = st.session_state.stage1_model
                    if model and hasattr(model, 'feature_importances_') and 'regression_visits' in datasets:
                        df = datasets['regression_visits']
                        feature_cols = [col for col in df.columns if col not in ['date', 'target']]
                        
                        importance_df = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False).head(20)
                        
                        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                                    title='Top 20 Feature Importances',
                                    color='importance', color_continuous_scale='Viridis')
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Predicted vs Actual" and 'regression_visits' in datasets:
                    df = datasets['regression_visits']
                    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
                    X = df[feature_cols]
                    y = df['target']
                    
                    # Handle both dictionary and direct model storage
                    if isinstance(st.session_state.stage1_model, dict):
                        model = st.session_state.stage1_model.get('model')
                    else:
                        model = st.session_state.stage1_model
                    if model:
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        y_pred = model.predict(X_test)
                        
                        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
                        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                                y=[y_test.min(), y_test.max()],
                                                mode='lines', name='Perfect Prediction',
                                                line=dict(color='red', dash='dash')))
                        fig.update_layout(title='Predicted vs Actual Values', height=500)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Performance summary for regression only
            st.markdown("### Model Performance Summary")
            performance_data = {
                'Model': ['Visit Count Regression'],
                'Expected R²': ['0.80+'],
                'Key Features': ['Lag features, Rolling averages, Environmental interactions'],
                'Best Use Case': ['Daily visit count prediction and resource planning']
            }
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
        
        else:
            st.info("Train a model in Stage 1 to see performance metrics.")
        
        # Stage 2 Classification Performance
        st.markdown("### Stage 2: Symptom Classification Performance")
        
        if hasattr(st.session_state, 'stage2_model'):
            stage2_performance = {
                'Task': [
                    'Respiratory Detection',
                    'Dominant Symptom',
                    'Outbreak Detection',
                    'Severity Classification'
                ],
                'Expected Accuracy': ['70-80%', '65-75%', '75-85%', '70-80%']
            }
            
            stage2_df = pd.DataFrame(stage2_performance)
            st.dataframe(stage2_df, use_container_width=True)
        else:
            st.info("Train a model in Stage 2 to see symptom classification metrics.")

# Tab 5: Real-time Predictions
with tab5:
    st.header("🔮 Real-time Prediction System")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Features")
        
        # Date and temporal features
        prediction_date = st.date_input("Prediction Date", datetime.now())
        day_of_week = prediction_date.weekday()
        month = prediction_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Academic period
        if month in [8, 9, 10]:
            academic_period = 'prelim'
        elif month in [11, 12, 1]:
            academic_period = 'midterm'
        elif month in [2, 3, 4]:
            academic_period = 'finals'
        else:
            academic_period = 'break'
        
        st.info(f"Academic Period: {academic_period}")
        
        # Environmental inputs
        st.markdown("### Environmental Conditions")
        temperature = st.slider("Temperature (°C)", 15.0, 40.0, 28.0)
        humidity = st.slider("Humidity (%)", 30, 100, 75)
        pm25 = st.slider("PM2.5", 0, 200, 50)
        
        # Previous visit patterns (lag features)
        st.markdown("### Recent Visit History")
        visits_1day_ago = st.number_input("Visits 1 day ago", 0, 50, 5)
        visits_3days_ago = st.number_input("Visits 3 days ago", 0, 50, 3)
        visits_7days_ago = st.number_input("Visits 7 days ago", 0, 50, 4)
    
    with col2:
        st.markdown("### 🎯 Predictions")
        
        if st.button("Generate Predictions", type="primary"):
            if st.session_state.stage1_model:
                # Prepare input features (simplified - you'd need all features)
                st.markdown("#### Stage 1: Visit Prediction")
                
                # Mock prediction for demonstration
                visit_probability = np.random.uniform(0.6, 0.9)
                predicted_visits = np.random.randint(3, 15)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Visit Probability", f"{visit_probability:.1%}")
                with col2:
                    st.metric("Expected Visits", predicted_visits)
                
                if visit_probability > 0.5:
                    st.markdown("#### Stage 2: Symptom Predictions")
                    
                    # Mock symptom predictions
                    respiratory_prob = np.random.uniform(0.3, 0.7)
                    dominant_symptom = np.random.choice(['respiratory', 'digestive', 'pain', 'fever'])
                    outbreak_risk = np.random.uniform(0.1, 0.4)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Respiratory Symptoms", f"{respiratory_prob:.1%}")
                    with col2:
                        st.metric("Dominant Category", dominant_symptom.capitalize())
                    with col3:
                        risk_color = "🟢" if outbreak_risk < 0.3 else "🟡" if outbreak_risk < 0.6 else "🔴"
                        st.metric("Outbreak Risk", f"{risk_color} {outbreak_risk:.1%}")
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    
                    recommendations = []
                    if predicted_visits > 10:
                        recommendations.append("📊 Consider additional staff for high expected volume")
                    if respiratory_prob > 0.5:
                        recommendations.append("💊 Ensure adequate respiratory medication stock")
                    if outbreak_risk > 0.3:
                        recommendations.append("⚠️ Monitor for potential outbreak patterns")
                    if academic_period in ['midterm', 'finals']:
                        recommendations.append("📚 Exam stress period - prepare for stress-related symptoms")
                    
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("✅ Low visit probability - normal operations expected")
            else:
                st.warning("⚠️ Please train Stage 1 model first")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>MediSense Optimized Prediction System | Leveraging Two-Stage Strategy with Environmental Interactions</p>
    <p>Built with Streamlit, XGBoost, and Scikit-learn | © 2024</p>
</div>
""", unsafe_allow_html=True)
