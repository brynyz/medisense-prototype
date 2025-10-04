"""
Test Script for Enhanced Visualizations
=======================================
This script creates sample data to test the visualization functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enhanced_visualizations import (
    create_regression_visualizations,
    create_classification_visualizations,
    create_feature_importance_visualization,
    create_summary_dashboard
)

def create_sample_data():
    """Create sample data for testing visualizations"""
    np.random.seed(42)
    
    # Sample regression data
    n_samples = 100
    y_test_reg = np.random.poisson(2, n_samples)  # Poisson for visit counts
    
    # Create predictions with some noise and bias
    y_pred_lr = y_test_reg + np.random.normal(0, 0.8, n_samples)
    y_pred_rf = y_test_reg + np.random.normal(0, 0.6, n_samples)
    y_pred_xgb = y_test_reg + np.random.normal(0, 0.7, n_samples)
    
    # Ensure non-negative predictions
    y_pred_lr = np.maximum(0, y_pred_lr)
    y_pred_rf = np.maximum(0, y_pred_rf)
    y_pred_xgb = np.maximum(0, y_pred_xgb)
    
    # Sample classification data
    n_classes = 4
    class_names = ['Respiratory', 'Digestive', 'Pain', 'Fever']
    y_test_clf = np.random.choice(n_classes, size=80)
    
    # Create predictions with some accuracy
    y_pred_baseline = y_test_clf.copy()
    # Add some errors
    error_indices = np.random.choice(len(y_pred_baseline), size=int(0.3 * len(y_pred_baseline)), replace=False)
    y_pred_baseline[error_indices] = np.random.choice(n_classes, size=len(error_indices))
    
    y_pred_xgb_clf = y_test_clf.copy()
    error_indices = np.random.choice(len(y_pred_xgb_clf), size=int(0.25 * len(y_pred_xgb_clf)), replace=False)
    y_pred_xgb_clf[error_indices] = np.random.choice(n_classes, size=len(error_indices))
    
    # Sample feature importance
    feature_names = [
        'temperature', 'humidity', 'pm25', 'pm10', 'academic_period',
        'day_of_week', 'respiratory_lag1', 'digestive_lag1', 'pain_lag1',
        'fever_lag1', 'weather_stress', 'symptom_diversity', 'total_visits_lag7'
    ]
    rf_importances = np.random.exponential(0.1, len(feature_names))
    rf_importances = rf_importances / rf_importances.sum()  # Normalize
    
    xgb_importances = rf_importances + np.random.normal(0, 0.02, len(feature_names))
    xgb_importances = np.maximum(0, xgb_importances)
    xgb_importances = xgb_importances / xgb_importances.sum()  # Normalize
    
    return {
        'regression': {
            'y_test': y_test_reg,
            'y_pred_lr': y_pred_lr,
            'y_pred_rf': y_pred_rf,
            'y_pred_xgb': y_pred_xgb
        },
        'classification': {
            'y_test': y_test_clf,
            'y_pred_baseline': y_pred_baseline,
            'y_pred_xgb': y_pred_xgb_clf,
            'class_names': class_names
        },
        'features': {
            'names': feature_names,
            'rf_importances': rf_importances,
            'xgb_importances': xgb_importances
        }
    }

def calculate_sample_metrics(data):
    """Calculate metrics for sample data"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Regression metrics
    reg_data = data['regression']
    regression_metrics = {}
    
    for model in ['lr', 'rf', 'xgb']:
        y_pred = reg_data[f'y_pred_{model}']
        y_test = reg_data['y_test']
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        within_one = np.mean(np.abs(y_test - y_pred) <= 1)
        
        regression_metrics[model] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'within_one': within_one
        }
    
    # Classification metrics
    clf_data = data['classification']
    classification_metrics = {}
    
    for model in ['baseline', 'xgb']:
        y_pred = clf_data[f'y_pred_{model}']
        y_test = clf_data['y_test']
        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        classification_metrics[model] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return regression_metrics, classification_metrics

def main():
    """Test all visualization functions"""
    print("🧪 Testing Enhanced Visualizations")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data()
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    regression_metrics, classification_metrics = calculate_sample_metrics(data)
    
    # Dataset info
    dataset_info = {
        'zero_visit_pct': 15.0,  # Sample percentage
        'num_classes': 4,
        'classification_samples': 80
    }
    
    output_path = 'medisense/backend/data/final/optimized/'
    
    print("🎨 Generating test visualizations...")
    
    # 1. Test Regression Visualizations
    print("   📈 Testing regression visualizations...")
    fig_reg = create_regression_visualizations(
        data['regression']['y_test'],
        data['regression']['y_pred_lr'],
        data['regression']['y_pred_rf'],
        data['regression']['y_pred_xgb'],
        regression_metrics
    )
    fig_reg.savefig(f'{output_path}test_regression_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: test_regression_analysis.png")
    plt.close(fig_reg)
    
    # 2. Test Classification Visualizations
    print("   🏥 Testing classification visualizations...")
    fig_clf = create_classification_visualizations(
        data['classification']['y_test'],
        data['classification']['y_pred_baseline'],
        data['classification']['y_pred_xgb'],
        data['classification']['class_names'],
        classification_metrics
    )
    fig_clf.savefig(f'{output_path}test_classification_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: test_classification_analysis.png")
    plt.close(fig_clf)
    
    # 3. Test Feature Importance
    print("   🔍 Testing feature importance visualizations...")
    fig_feat = create_feature_importance_visualization(
        data['features']['names'],
        data['features']['rf_importances'],
        data['features']['xgb_importances']
    )
    fig_feat.savefig(f'{output_path}test_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: test_feature_importance.png")
    plt.close(fig_feat)
    
    # 4. Test Summary Dashboard
    print("   📊 Testing summary dashboard...")
    fig_summary = create_summary_dashboard(regression_metrics, classification_metrics, dataset_info)
    fig_summary.savefig(f'{output_path}test_evaluation_summary.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: test_evaluation_summary.png")
    plt.close(fig_summary)
    
    print(f"\n✅ All test visualizations completed!")
    print(f"   📁 Check the output directory: {output_path}")
    print(f"   🎯 Files created with 'test_' prefix")
    
    # Print sample metrics
    print(f"\n📋 Sample Metrics:")
    print(f"   Regression R²: {regression_metrics['rf']['r2']:.3f}")
    print(f"   Classification Accuracy: {classification_metrics['baseline']['accuracy']:.3f}")

if __name__ == "__main__":
    main()
