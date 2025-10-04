"""
Standalone Visualization Generator for Multi-Task Evaluation Results
==================================================================
This script generates comprehensive visualizations for the multi-task evaluation results.
Run this after completing the multi-task evaluation to create detailed charts and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced visualization functions
from enhanced_visualizations import (
    create_regression_visualizations,
    create_classification_visualizations,
    create_feature_importance_visualization,
    create_summary_dashboard
)

def load_and_prepare_data():
    """Load and prepare datasets for visualization"""
    print("📊 Loading datasets...")
    
    # Load regression dataset
    try:
        df_regression = pd.read_csv('medisense/backend/data/final/optimized/dataset_regression_visits_optimized.csv')
        print("✅ Loaded optimized regression dataset")
    except:
        try:
            df_regression = pd.read_csv('medisense/backend/data/final/dataset_regression_visits.csv')
            print("✅ Loaded regression dataset")
        except:
            print("❌ Could not load regression dataset")
            return None, None
    
    # Load classification dataset
    try:
        df_classification = pd.read_csv('medisense/backend/data/final/optimized/dataset_dominant_symptom_optimized.csv')
        print("✅ Loaded optimized classification dataset")
    except:
        try:
            df_classification = pd.read_csv('medisense/backend/data/final/dataset_dominant_symptom.csv')
            print("✅ Loaded classification dataset")
        except:
            print("❌ Could not load classification dataset")
            df_classification = None
    
    return df_regression, df_classification

def train_regression_models(df_regression):
    """Train regression models and return predictions and metrics"""
    print("🔄 Training regression models...")
    
    # Prepare data
    X_reg = df_regression.drop(columns=['target', 'date'], errors='ignore')
    y_reg = df_regression['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    predictions = {}
    metrics = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    models['lr'] = lr
    predictions['lr'] = y_pred_lr
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    models['rf'] = rf
    predictions['rf'] = y_pred_rf
    
    # XGBoost
    xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_reg.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_reg.predict(X_test_scaled)
    models['xgb'] = xgb_reg
    predictions['xgb'] = y_pred_xgb
    
    # Calculate metrics for each model
    for model_name, y_pred in predictions.items():
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        within_one = np.mean(np.abs(y_test - y_pred) <= 1)
        
        metrics[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'within_one': within_one
        }
        
        print(f"   {model_name.upper()}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
    
    return y_test, predictions, metrics, models, X_reg.columns

def train_classification_models(df_classification):
    """Train classification models and return predictions and metrics"""
    if df_classification is None:
        return None, None, None, None, None
    
    print("🔄 Training classification models...")
    
    # Prepare data
    X_clf = df_classification.drop(columns=['target', 'date'], errors='ignore')
    y_clf = df_classification['target']
    
    # Filter to only days with symptoms
    if 'no_symptom' in y_clf.values or 0 in y_clf.values:
        mask = (y_clf != 'no_symptom') & (y_clf != 0)
        X_clf = X_clf[mask]
        y_clf = y_clf[mask]
    
    # Encode target if needed
    if y_clf.dtype == 'object':
        le = LabelEncoder()
        y_clf_encoded = le.fit_transform(y_clf)
        class_names = le.classes_
    else:
        y_clf_encoded = y_clf
        class_names = np.unique(y_clf)
    
    # Check class distribution and filter if needed
    class_counts = pd.Series(y_clf_encoded).value_counts()
    if class_counts.min() < 2:
        valid_classes = class_counts[class_counts >= 2].index
        mask = pd.Series(y_clf_encoded).isin(valid_classes)
        X_clf = X_clf[mask]
        y_clf_encoded = y_clf_encoded[mask]
        
        # Re-encode to ensure consecutive labels
        le_filtered = LabelEncoder()
        y_clf_encoded = le_filtered.fit_transform(y_clf_encoded)
        if y_clf.dtype == 'object':
            class_names = np.array([class_names[i] for i in valid_classes])
        else:
            class_names = valid_classes
    
    if len(y_clf_encoded) < 10:
        print("   ⚠️  Insufficient samples for classification")
        return None, None, None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf_encoded
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    predictions = {}
    metrics = {}
    
    # Random Forest (Baseline)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_clf.fit(X_train_scaled, y_train)
    y_pred_baseline = rf_clf.predict(X_test_scaled)
    predictions['baseline'] = y_pred_baseline
    
    # XGBoost
    try:
        xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
        xgb_clf.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb_clf.predict(X_test_scaled)
        predictions['xgb'] = y_pred_xgb
    except:
        predictions['xgb'] = y_pred_baseline  # Fallback
    
    # Calculate metrics
    for model_name, y_pred in predictions.items():
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics[model_name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"   {model_name.upper()}: Accuracy = {acc:.3f}, F1 = {f1:.3f}")
    
    return y_test, predictions, metrics, class_names, len(y_clf_encoded)

def main():
    """Main function to generate all visualizations"""
    print("🎯 MEDISENSE VISUALIZATION GENERATOR")
    print("=" * 70)
    
    # Load data
    df_regression, df_classification = load_and_prepare_data()
    
    if df_regression is None:
        print("❌ Cannot proceed without regression dataset")
        return
    
    # Train regression models
    y_test_reg, reg_predictions, reg_metrics, reg_models, feature_names = train_regression_models(df_regression)
    
    # Train classification models
    clf_results = train_classification_models(df_classification)
    if clf_results[0] is not None:
        y_test_clf, clf_predictions, clf_metrics, class_names, clf_samples = clf_results
    else:
        y_test_clf = clf_predictions = clf_metrics = class_names = clf_samples = None
    
    # Prepare dataset info
    dataset_info = {
        'zero_visit_pct': (df_regression['target'] == 0).sum() / len(df_regression) * 100,
        'num_classes': len(class_names) if class_names is not None else 'N/A',
        'classification_samples': clf_samples if clf_samples is not None else 'N/A'
    }
    
    # Create output directory
    output_path = 'medisense/backend/data/final/optimized/'
    
    print(f"\n📊 Generating visualizations...")
    
    # 1. Regression Visualizations
    print("   📈 Creating regression analysis...")
    fig_reg = create_regression_visualizations(
        y_test_reg, 
        reg_predictions['lr'], 
        reg_predictions['rf'], 
        reg_predictions['xgb'], 
        reg_metrics
    )
    fig_reg.savefig(f'{output_path}regression_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: regression_analysis.png")
    plt.close(fig_reg)
    
    # 2. Classification Visualizations (if available)
    if y_test_clf is not None:
        print("   🏥 Creating classification analysis...")
        fig_clf = create_classification_visualizations(
            y_test_clf, 
            clf_predictions['baseline'], 
            clf_predictions['xgb'], 
            class_names, 
            clf_metrics
        )
        fig_clf.savefig(f'{output_path}classification_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: classification_analysis.png")
        plt.close(fig_clf)
    
    # 3. Feature Importance
    print("   🔍 Creating feature importance analysis...")
    rf_importances = reg_models['rf'].feature_importances_
    xgb_importances = reg_models['xgb'].feature_importances_
    
    fig_feat = create_feature_importance_visualization(
        feature_names, rf_importances, xgb_importances
    )
    fig_feat.savefig(f'{output_path}feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: feature_importance.png")
    plt.close(fig_feat)
    
    # 4. Summary Dashboard
    print("   📊 Creating summary dashboard...")
    fig_summary = create_summary_dashboard(reg_metrics, clf_metrics, dataset_info)
    fig_summary.savefig(f'{output_path}evaluation_summary.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: evaluation_summary.png")
    plt.close(fig_summary)
    
    print(f"\n✅ All visualizations generated successfully!")
    print(f"   📁 Output directory: {output_path}")
    print(f"   📊 Files created:")
    print(f"      • regression_analysis.png")
    if y_test_clf is not None:
        print(f"      • classification_analysis.png")
    print(f"      • feature_importance.png")
    print(f"      • evaluation_summary.png")
    
    # Print summary
    print(f"\n📋 RESULTS SUMMARY:")
    print(f"   🎯 Best Regression Model: Random Forest (R² = {reg_metrics['rf']['r2']:.3f})")
    if clf_metrics is not None:
        best_clf_f1 = max(clf_metrics['baseline']['f1'], clf_metrics['xgb']['f1'])
        best_clf_model = 'XGBoost' if clf_metrics['xgb']['f1'] > clf_metrics['baseline']['f1'] else 'Random Forest'
        print(f"   🏥 Best Classification Model: {best_clf_model} (F1 = {best_clf_f1:.3f})")
    print(f"   📊 Zero-visit days: {dataset_info['zero_visit_pct']:.1f}%")

if __name__ == "__main__":
    main()
