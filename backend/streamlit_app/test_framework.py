import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_symptom_categories(df):
    """Analyze symptom category distribution and quality"""
    st.subheader("🔍 Symptom Category Analysis")
    
    # Find symptom columns
    symptom_cols = [col for col in df.columns if any(symptom in col.lower() for symptom in 
                   ['respiratory', 'digestive', 'pain', 'fever', 'neurological', 'skin', 'cardiovascular'])]
    
    if not symptom_cols:
        st.error("❌ No symptom category columns found!")
        st.write("Available columns:", df.columns.tolist())
        return None
    
    st.write(f"Found symptom columns: {symptom_cols}")
    
    # Analyze symptom data
    symptom_data = df[symptom_cols].fillna(0)
    
    # Show distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Symptom Category Sums:**")
        symptom_sums = symptom_data.sum().sort_values(ascending=False)
        st.bar_chart(symptom_sums)
        
    with col2:
        st.write("**Non-zero counts:**")
        non_zero_counts = (symptom_data > 0).sum()
        st.write(non_zero_counts)
    
    # Check for dominant categories
    st.write("**Dominant symptom per day:**")
    dominant_symptoms = symptom_data.idxmax(axis=1)
    dominant_counts = dominant_symptoms.value_counts()
    
    st.write(dominant_counts)
    
    # Check for days with no symptoms
    no_symptoms = (symptom_data.sum(axis=1) == 0).sum()
    st.write(f"Days with no symptoms: {no_symptoms} ({no_symptoms/len(df)*100:.1f}%)")
    
    return symptom_cols, symptom_data, dominant_symptoms

def improved_symptom_category_preparation(df):
    """Improved symptom category preparation"""
    st.subheader("🔧 Improved Symptom Category Preparation")
    
    # Analyze current data
    result = analyze_symptom_categories(df)
    if result is None:
        return None, None
    
    symptom_cols, symptom_data, dominant_symptoms = result
    
    # Strategy selection
    strategy = st.selectbox(
        "Choose target creation strategy:",
        [
            "Dominant Symptom (Original)",
            "Binary Presence/Absence", 
            "High/Low Symptom Days",
            "Respiratory vs Non-Respiratory",
            "Top 3 Categories Only"
        ]
    )
    
    if strategy == "Dominant Symptom (Original)":
        # Original approach - but handle zero days better
        targets = dominant_symptoms.copy()
        
        # Handle days with no symptoms
        zero_symptom_mask = symptom_data.sum(axis=1) == 0
        targets[zero_symptom_mask] = 'no_symptoms'
        
        st.write(f"Target distribution:")
        st.write(targets.value_counts())
        
    elif strategy == "Binary Presence/Absence":
        # Simpler binary classification
        targets = (symptom_data.sum(axis=1) > 0).astype(int)
        targets = targets.map({0: 'no_symptoms', 1: 'has_symptoms'})
        
    elif strategy == "High/Low Symptom Days":
        # Based on total symptom count
        total_symptoms = symptom_data.sum(axis=1)
        threshold = total_symptoms.quantile(0.5)
        targets = (total_symptoms > threshold).astype(int)
        targets = targets.map({0: 'low_symptoms', 1: 'high_symptoms'})
        
    elif strategy == "Respiratory vs Non-Respiratory":
        # Focus on respiratory vs everything else
        respiratory_cols = [col for col in symptom_cols if 'respiratory' in col.lower()]
        if respiratory_cols:
            respiratory_sum = df[respiratory_cols].sum(axis=1)
            other_sum = symptom_data.sum(axis=1) - respiratory_sum
            
            conditions = [
                (respiratory_sum > other_sum),
                (other_sum > respiratory_sum),
                (symptom_data.sum(axis=1) == 0)
            ]
            choices = ['respiratory', 'non_respiratory', 'no_symptoms']
            targets = pd.Series(np.select(conditions, choices, default='mixed'), index=df.index)
        else:
            st.error("No respiratory columns found!")
            return None, None
            
    elif strategy == "Top 3 Categories Only":
        # Only use top 3 most common categories
        top_3_symptoms = symptom_data.sum().nlargest(3).index
        
        # Filter data to only top 3
        top_3_data = symptom_data[top_3_symptoms]
        targets = top_3_data.idxmax(axis=1)
        
        # Handle days with no symptoms in top 3
        zero_top3_mask = top_3_data.sum(axis=1) == 0
        targets[zero_top3_mask] = 'other_or_none'
    
    # Convert to numeric labels
    le = LabelEncoder()
    targets_numeric = le.fit_transform(targets)
    
    st.write(f"**Final target distribution:**")
    target_dist = pd.Series(targets).value_counts()
    st.write(target_dist)
    
    # Check class balance
    min_class_size = target_dist.min()
    max_class_size = target_dist.max()
    imbalance_ratio = max_class_size / min_class_size
    
    st.write(f"**Class imbalance ratio:** {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        st.warning("⚠️ Severe class imbalance detected! Consider balancing techniques.")
    
    return targets_numeric, le.classes_

def enhanced_feature_engineering(df, target_col):
    """Enhanced feature engineering for symptom prediction"""
    st.subheader("⚙️ Enhanced Feature Engineering")
    
    df_enhanced = df.copy()
    
    # Feature engineering options
    options = st.multiselect(
        "Select feature engineering techniques:",
        [
            "Lag Features", 
            "Rolling Averages",
            "Weather Interactions", 
            "Pollution Risk Scores",
            "Academic Period Encoding",
            "Seasonal Features"
        ],
        default=["Lag Features", "Rolling Averages", "Weather Interactions"]
    )
    
    if "Lag Features" in options:
        # Add lag features for visit counts and environmental data
        lag_cols = ['visit_count', 'temperature_2m_mean', 'pm2_5_mean', 'pm10_mean']
        for col in lag_cols:
            if col in df_enhanced.columns:
                for lag in [1, 3, 7]:
                    df_enhanced[f'{col}_lag_{lag}'] = df_enhanced[col].shift(lag)
    
    if "Rolling Averages" in options:
        # Rolling averages for environmental data
        env_cols = [col for col in df_enhanced.columns if any(x in col.lower() for x in 
                   ['temperature', 'humidity', 'pm2_5', 'pm10', 'ozone'])]
        for col in env_cols:
            if col in df_enhanced.columns:
                df_enhanced[f'{col}_rolling_7'] = df_enhanced[col].rolling(7, min_periods=1).mean()
                df_enhanced[f'{col}_rolling_14'] = df_enhanced[col].rolling(14, min_periods=1).mean()
    
    if "Weather Interactions" in options:
        # Create interaction features
        if all(col in df_enhanced.columns for col in ['temperature_2m_mean', 'relative_humidity_2m_mean']):
            df_enhanced['heat_index'] = df_enhanced['temperature_2m_mean'] * df_enhanced['relative_humidity_2m_mean'] / 100
        
        if all(col in df_enhanced.columns for col in ['pm2_5_mean', 'temperature_2m_mean']):
            df_enhanced['pollution_temp_interaction'] = df_enhanced['pm2_5_mean'] * df_enhanced['temperature_2m_mean']
    
    if "Pollution Risk Scores" in options:
        # Create pollution risk composite scores
        pollution_cols = [col for col in df_enhanced.columns if any(x in col.lower() for x in 
                         ['pm2_5', 'pm10', 'ozone', 'nitrogen_dioxide'])]
        if pollution_cols:
            # Normalize and combine
            pollution_data = df_enhanced[pollution_cols].fillna(0)
            pollution_normalized = (pollution_data - pollution_data.mean()) / pollution_data.std()
            df_enhanced['pollution_risk_score'] = pollution_normalized.mean(axis=1)
    
    if "Academic Period Encoding" in options:
        # Better academic period encoding
        if 'academic_period' in df_enhanced.columns:
            # One-hot encode academic periods
            academic_dummies = pd.get_dummies(df_enhanced['academic_period'], prefix='academic')
            df_enhanced = pd.concat([df_enhanced, academic_dummies], axis=1)
    
    if "Seasonal Features" in options:
        # Add seasonal features
        if 'date' in df_enhanced.columns:
            df_enhanced['date'] = pd.to_datetime(df_enhanced['date'])
            df_enhanced['month'] = df_enhanced['date'].dt.month
            df_enhanced['day_of_year'] = df_enhanced['date'].dt.dayofyear
            df_enhanced['season'] = df_enhanced['month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            })
            season_dummies = pd.get_dummies(df_enhanced['season'], prefix='season')
            df_enhanced = pd.concat([df_enhanced, season_dummies], axis=1)
    
    # Remove rows with NaN values created by lag features
    df_enhanced = df_enhanced.dropna()
    
    st.write(f"Enhanced dataset shape: {df_enhanced.shape}")
    st.write(f"Added {df_enhanced.shape[1] - df.shape[1]} new features")
    
    return df_enhanced

def test_improved_model(df):
    """Test improved symptom category prediction"""
    st.header("🧪 Improved Model Testing")
    
    # Step 1: Improved target preparation
    targets, class_names = improved_symptom_category_preparation(df)
    if targets is None:
        return
    
    # Step 2: Enhanced feature engineering
    df_enhanced = enhanced_feature_engineering(df, targets)
    
    # Align targets with enhanced dataframe
    targets_aligned = targets[:len(df_enhanced)]
    
    # Step 3: Feature selection
    st.subheader("🎯 Feature Selection")
    
    # Exclude columns
    exclude_cols = [
        'date', 'date_cleaned', 'week_start_date', 'normalized_symptoms', 
        'course_mapped', 'gender', 'symptoms', 'academic_period'
    ]
    
    # Get symptom columns to exclude
    symptom_cols = [col for col in df_enhanced.columns if any(symptom in col.lower() for symptom in 
                   ['respiratory', 'digestive', 'pain', 'fever', 'neurological', 'skin', 'cardiovascular'])]
    exclude_cols.extend(symptom_cols)
    
    # Select features
    feature_cols = [col for col in df_enhanced.columns 
                   if col not in exclude_cols and df_enhanced[col].dtype in ['int64', 'float64']]
    
    st.write(f"Selected {len(feature_cols)} features for training")
    
    with st.expander("View selected features"):
        st.write(feature_cols)
    
    # Step 4: Model training with improvements
    X = df_enhanced[feature_cols].fillna(0)
    y = targets_aligned
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model selection
    model_type = st.selectbox(
        "Select model:",
        ["Random Forest", "XGBoost", "Random Forest + Balanced"]
    )
    
    if st.button("🚀 Train Improved Model"):
        with st.spinner("Training improved model..."):
            
            if model_type == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            elif model_type == "XGBoost":
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
            else:  # Random Forest + Balanced
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',  # Handle imbalanced classes
                    random_state=42
                )
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluate
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Display results
            st.subheader("📊 Improved Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1 Score", f"{f1:.4f}")
            
            # Confusion Matrix
            st.subheader("🔥 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=class_names,
                y=class_names,
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("⭐ Feature Importance")
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(20)
                
                fig = px.bar(
                    importance_df, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title="Top 20 Feature Importances"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🧪 Symptom Category Prediction Test Framework")
    st.markdown("Diagnose and improve your symptom category prediction model")
    
    # Load data
    st.sidebar.header("📁 Data Loading")
    data_source = st.sidebar.selectbox(
        "Select Dataset:",
        ["daily_complete.csv", "daily_visits_only.csv", "Upload Custom"]
    )
    
    # Load data logic here (similar to your main app)
    # For now, assuming data is loaded
    if st.sidebar.button("Load Test Data"):
        # Mock data loading - replace with actual loading
        st.info("Please integrate with your data loading logic from the main app")
        
        # You can integrate this with your existing app.py data loading
        # df = load_data(data_path)
        # if df is not None:
        #     test_improved_model(df)

if __name__ == "__main__":
    main()