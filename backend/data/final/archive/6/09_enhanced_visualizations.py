"""
Enhanced Visualizations for Multi-Task Model Evaluation
======================================================
This script creates comprehensive visualizations for the multi-task evaluation results
including regression, classification, and comparative analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_regression_visualizations(y_test, y_pred_lr, y_pred_rf, y_pred_xgb, metrics):
    """Create comprehensive regression visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Actual vs Predicted - Random Forest (Best Model)
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred_rf, alpha=0.6, color='#2ecc71', s=50)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Visit Count', fontsize=12)
    ax1.set_ylabel('Predicted Visit Count', fontsize=12)
    ax1.set_title(f'Random Forest: Actual vs Predicted\nR² = {metrics["rf"]["r2"]:.3f}', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals Analysis
    ax2 = axes[0, 1]
    residuals = y_test - y_pred_rf
    ax2.scatter(y_pred_rf, residuals, alpha=0.6, color='#e74c3c', s=50)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals Distribution
    ax3 = axes[0, 2]
    ax3.hist(residuals, bins=30, edgecolor='black', color='#9b59b6', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'Residual Distribution\nMAE = {metrics["rf"]["mae"]:.3f}', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Comparison - R² Scores
    ax4 = axes[1, 0]
    models = ['Linear\nRegression', 'Random\nForest', 'XGBoost']
    r2_scores = [metrics['lr']['r2'], metrics['rf']['r2'], metrics['xgb']['r2']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax4.bar(models, r2_scores, color=colors, alpha=0.8)
    ax4.set_ylabel('R² Score', fontsize=12)
    ax4.set_title('Model Comparison: R² Scores', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, max(r2_scores) * 1.1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, r2_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. RMSE Comparison
    ax5 = axes[1, 1]
    rmse_scores = [metrics['lr']['rmse'], metrics['rf']['rmse'], metrics['xgb']['rmse']]
    bars = ax5.bar(models, rmse_scores, color=colors, alpha=0.8)
    ax5.set_ylabel('RMSE', fontsize=12)
    ax5.set_title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, rmse_scores):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Accuracy Within ±1 Visit
    ax6 = axes[1, 2]
    within_one = [metrics['lr']['within_one'], metrics['rf']['within_one'], metrics['xgb']['within_one']]
    bars = ax6.bar(models, within_one, color=colors, alpha=0.8)
    ax6.set_ylabel('Accuracy Within ±1 Visit', fontsize=12)
    ax6.set_title('Practical Accuracy Comparison', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, within_one):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{score:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Regression Analysis: Visit Count Prediction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_classification_visualizations(y_test, y_pred_baseline, y_pred_xgb, class_names, metrics):
    """Create comprehensive classification visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Confusion Matrix - Baseline
    ax1 = axes[0, 0]
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title(f'Random Forest: Confusion Matrix\nAccuracy = {metrics["baseline"]["accuracy"]:.3f}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    
    # 2. Confusion Matrix - XGBoost
    ax2 = axes[0, 1]
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title(f'XGBoost: Confusion Matrix\nAccuracy = {metrics["xgb"]["accuracy"]:.3f}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    
    # 3. Class Distribution
    ax3 = axes[0, 2]
    class_counts = pd.Series(y_test).value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    
    bars = ax3.bar(range(len(class_counts)), class_counts.values, color=colors, alpha=0.8)
    ax3.set_xlabel('Class', fontsize=12)
    ax3.set_ylabel('Sample Count', fontsize=12)
    ax3.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, class_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='bold')
    
    # 4. Model Performance Comparison
    ax4 = axes[1, 0]
    models = ['Random Forest', 'XGBoost']
    accuracy_scores = [metrics['baseline']['accuracy'], metrics['xgb']['accuracy']]
    f1_scores = [metrics['baseline']['f1'], metrics['xgb']['f1']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='#3498db', alpha=0.8)
    bars2 = ax4.bar(x + width/2, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Classification Model Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Precision-Recall by Class
    ax5 = axes[1, 1]
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_baseline)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax5.bar(x - width, precision, width, label='Precision', color='#2ecc71', alpha=0.8)
    bars2 = ax5.bar(x, recall, width, label='Recall', color='#f39c12', alpha=0.8)
    bars3 = ax5.bar(x + width, f1, width, label='F1-Score', color='#9b59b6', alpha=0.8)
    
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('Per-Class Performance (Random Forest)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(class_names, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 1])
    
    # 6. Sample Support by Class
    ax6 = axes[1, 2]
    bars = ax6.bar(range(len(support)), support, color=colors, alpha=0.8)
    ax6.set_xlabel('Class', fontsize=12)
    ax6.set_ylabel('Sample Count', fontsize=12)
    ax6.set_title('Test Sample Support by Class', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(len(class_names)))
    ax6.set_xticklabels(class_names, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, support):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Classification Analysis: Dominant Symptom Prediction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_feature_importance_visualization(feature_names, importances_rf, importances_xgb=None):
    """Create feature importance visualization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Random Forest Feature Importance
    ax1 = axes[0]
    top_features = 15
    indices = np.argsort(importances_rf)[-top_features:]
    
    ax1.barh(range(len(indices)), importances_rf[indices], color='#2ecc71', alpha=0.8)
    ax1.set_yticks(range(len(indices)))
    ax1.set_yticklabels([feature_names[i] for i in indices])
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.set_title('Top 15 Feature Importances\n(Random Forest)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # XGBoost Feature Importance (if available)
    if importances_xgb is not None:
        ax2 = axes[1]
        indices_xgb = np.argsort(importances_xgb)[-top_features:]
        
        ax2.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], color='#e74c3c', alpha=0.8)
        ax2.set_yticks(range(len(indices_xgb)))
        ax2.set_yticklabels([feature_names[i] for i in indices_xgb])
        ax2.set_xlabel('Feature Importance', fontsize=12)
        ax2.set_title('Top 15 Feature Importances\n(XGBoost)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
    else:
        ax2 = axes[1]
        ax2.text(0.5, 0.5, 'XGBoost feature importance\nnot available', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
    
    plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_summary_dashboard(regression_metrics, classification_metrics, dataset_info):
    """Create a summary dashboard with key insights"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Key Metrics Summary
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    summary_text = f"""
    📊 MULTI-TASK EVALUATION SUMMARY
    
    🎯 REGRESSION (Visit Count Prediction):
    • Best Model: Random Forest
    • R² Score: {regression_metrics['rf']['r2']:.3f}
    • RMSE: {regression_metrics['rf']['rmse']:.2f} visits
    • Within ±1 visit: {regression_metrics['rf']['within_one']:.1%}
    • Zero-visit days: {dataset_info.get('zero_visit_pct', 'N/A')}%
    
    🏥 CLASSIFICATION (Dominant Symptom):
    • Best Model: {'XGBoost' if classification_metrics['xgb']['f1'] > classification_metrics['baseline']['f1'] else 'Random Forest'}
    • Accuracy: {max(classification_metrics['baseline']['accuracy'], classification_metrics['xgb']['accuracy']):.3f}
    • F1-Score: {max(classification_metrics['baseline']['f1'], classification_metrics['xgb']['f1']):.3f}
    • Classes: {dataset_info.get('num_classes', 'N/A')} symptom categories
    • Dataset size: {dataset_info.get('classification_samples', 'N/A')} samples
    """
    
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 2. Performance Comparison
    ax2 = axes[0, 1]
    tasks = ['Regression\n(R²)', 'Classification\n(Accuracy)']
    scores = [regression_metrics['rf']['r2'], 
              max(classification_metrics['baseline']['accuracy'], classification_metrics['xgb']['accuracy'])]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax2.bar(tasks, scores, color=colors, alpha=0.8)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Task Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Challenge Analysis
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    challenges_text = f"""
    ⚠️ KEY CHALLENGES IDENTIFIED:
    
    • Severe Class Imbalance:
      - {dataset_info.get('zero_visit_pct', 85.8):.1f}% zero-visit days in regression
      - Uneven symptom distribution in classification
    
    • Limited Data for Rare Events:
      - Some symptom classes have very few samples
      - Affects model generalization
    
    • Temporal Dependencies:
      - Medical trends have seasonal patterns
      - Requires careful time-series validation
    
    • Feature Engineering Needs:
      - Lag features crucial for prediction
      - Environmental factors important
    """
    
    ax3.text(0.05, 0.95, challenges_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # 4. Recommendations
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    recommendations_text = f"""
    💡 RECOMMENDATIONS:
    
    🔧 Technical Improvements:
    • Use ensemble methods (RF/XGBoost)
    • Implement two-stage regression model
    • Apply SMOTE for class balancing
    • Add more lag features
    
    📊 Data Collection:
    • Gather more samples for rare symptoms
    • Collect longer time series data
    • Include more environmental variables
    
    🚀 Deployment Strategy:
    • Start with binary classification
    • Use regression for capacity planning
    • Implement proper monitoring
    • Regular model retraining
    """
    
    ax4.text(0.05, 0.95, recommendations_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.suptitle('Multi-Task Evaluation: Summary & Insights', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("📊 Enhanced Visualization Script Ready!")
    print("Import this module and call the visualization functions with your data.")
