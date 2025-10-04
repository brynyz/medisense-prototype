"""
Model Training with SMOTE Implementation
=========================================
This script properly implements SMOTE to handle class imbalance:
1. Splits data into train-test BEFORE any resampling
2. Establishes baseline model performance
3. Applies SMOTE only to training data
4. Compares baseline vs SMOTE-enhanced models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("🤖 MEDISENSE MODEL TRAINING WITH SMOTE")
print("=" * 70)
print("Proper implementation to prevent data leakage")
print("=" * 70)

# ==========================================
# STEP 1: LOAD AND PREPARE DATA 📍
print("\n" + "="*70)
print("STEP 1: DATA LOADING AND PREPARATION ")
print("="*70)

# Load the optimized dataset
print("📂 Loading optimized dataset...")
# Try multiple possible locations
dataset_path = 'medisense/backend/data/final/dataset_binary_visits.csv'
try:
    df = pd.read_csv(dataset_path)
    print(f"   ✅ Loaded dataset: {dataset_path}")
except:
    try:
        dataset_path = 'medisense/backend/data/final/optimized/daily_optimized.csv'
        df = pd.read_csv(dataset_path)
        print(f"   ✅ Loaded dataset: {dataset_path}")
    except:
        dataset_path = 'medisense/backend/data/processed/daily_optimized.csv'
        df = pd.read_csv(dataset_path)
        print(f"   ✅ Loaded dataset: {dataset_path}")

print(f"   Shape: {df.shape}")

# Separate features and target
X = df.drop(columns=['target', 'date'], errors='ignore')
y = df['target']

print(f"\n📊 Dataset Statistics:")
print(f"   Total samples: {len(y)}")
print(f"   Features: {X.shape[1]}")
print(f"\n   Class distribution:")
class_counts = y.value_counts()
for class_val, count in class_counts.items():
    percentage = (count / len(y)) * 100
    print(f"      Class {class_val}: {count} samples ({percentage:.1f}%)")

# Calculate imbalance ratio
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\n   Imbalance ratio: {imbalance_ratio:.2f}:1")

# ==========================================
# CRITICAL: TRAIN-TEST SPLIT (BEFORE SMOTE!)
# ==========================================

print("\n" + "="*70)
print("TRAIN-TEST SPLIT (CRITICAL STEP!) 🔒")
print("="*70)

# Perform train-test split BEFORE any resampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain class distribution in both sets
)

print(f"\n✅ Data split completed:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Check class distribution in splits
print(f"\n   Training set class distribution:")
train_counts = y_train.value_counts()
for class_val, count in train_counts.items():
    percentage = (count / len(y_train)) * 100
    print(f"      Class {class_val}: {count} samples ({percentage:.1f}%)")

print(f"\n   Test set class distribution:")
test_counts = y_test.value_counts()
for class_val, count in test_counts.items():
    percentage = (count / len(y_test)) * 100
    print(f"      Class {class_val}: {count} samples ({percentage:.1f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler fitted on training data

print(f"\n✅ Features standardized")

# ==========================================
# STEP 2: BASELINE MODEL (NO SMOTE) 📊
# ==========================================

print("\n" + "="*70)
print("STEP 2: BASELINE MODEL (WITHOUT SMOTE) 📊")
print("="*70)

# Train baseline Random Forest model on imbalanced data
print("\n🌲 Training Random Forest baseline on imbalanced data...")
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Use class weights to partially address imbalance
)

rf_baseline.fit(X_train_scaled, y_train)

# Evaluate baseline model
y_pred_baseline = rf_baseline.predict(X_test_scaled)
y_pred_proba_baseline = rf_baseline.predict_proba(X_test_scaled)

# Calculate metrics for baseline
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
precision_baseline = precision_score(y_test, y_pred_baseline, average='weighted')
recall_baseline = recall_score(y_test, y_pred_baseline, average='weighted')
f1_baseline = f1_score(y_test, y_pred_baseline, average='weighted')

# For binary classification, calculate AUC-ROC
if len(np.unique(y)) == 2:
    auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline[:, 1])
else:
    auc_baseline = None

print(f"\n📈 BASELINE MODEL PERFORMANCE:")
print(f"   Accuracy: {accuracy_baseline:.4f}")
print(f"   Precision: {precision_baseline:.4f}")
print(f"   Recall: {recall_baseline:.4f}")
print(f"   F1-Score: {f1_baseline:.4f}")
if auc_baseline:
    print(f"   AUC-ROC: {auc_baseline:.4f}")

print(f"\n📋 Baseline Classification Report:")
print(classification_report(y_test, y_pred_baseline))

print(f"🔲 Baseline Confusion Matrix:")
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
print(cm_baseline)

# ==========================================
# STEP 3: APPLY SMOTE TO TRAINING DATA ⚖️
# ==========================================

print("\n" + "="*70)
print("STEP 3: APPLYING SMOTE TO TRAINING DATA ONLY ⚖️")
print("="*70)

# Initialize SMOTE
smote = SMOTE(
    random_state=42,
    sampling_strategy='auto',  # Balance all classes
    k_neighbors=5
)

print(f"\n🔄 Applying SMOTE to training data...")
print(f"   Original training samples: {X_train_scaled.shape[0]}")

# Apply SMOTE only to training data
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"   Resampled training samples: {X_resampled.shape[0]}")
print(f"   Synthetic samples created: {X_resampled.shape[0] - X_train_scaled.shape[0]}")

# Check new class distribution
print(f"\n   Resampled class distribution:")
resampled_counts = pd.Series(y_resampled).value_counts()
for class_val, count in resampled_counts.items():
    percentage = (count / len(y_resampled)) * 100
    print(f"      Class {class_val}: {count} samples ({percentage:.1f}%)")

# ==========================================
# STEP 4: TRAIN MODEL ON RESAMPLED DATA ✅
# ==========================================

print("\n" + "="*70)
print("STEP 4: TRAINING MODEL ON SMOTE-RESAMPLED DATA ✅")
print("="*70)

# Train new Random Forest model on balanced data
print("\n🌲 Training Random Forest on SMOTE-balanced data...")
rf_smote = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
    # Note: No class_weight needed since data is now balanced
)

rf_smote.fit(X_resampled, y_resampled)

# Evaluate SMOTE model on ORIGINAL TEST SET (not resampled!)
y_pred_smote = rf_smote.predict(X_test_scaled)
y_pred_proba_smote = rf_smote.predict_proba(X_test_scaled)

# Calculate metrics for SMOTE model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote, average='weighted')
recall_smote = recall_score(y_test, y_pred_smote, average='weighted')
f1_smote = f1_score(y_test, y_pred_smote, average='weighted')

# For binary classification, calculate AUC-ROC
if len(np.unique(y)) == 2:
    auc_smote = roc_auc_score(y_test, y_pred_proba_smote[:, 1])
else:
    auc_smote = None

print(f"\n📈 SMOTE MODEL PERFORMANCE:")
print(f"   Accuracy: {accuracy_smote:.4f}")
print(f"   Precision: {precision_smote:.4f}")
print(f"   Recall: {recall_smote:.4f}")
print(f"   F1-Score: {f1_smote:.4f}")
if auc_smote:
    print(f"   AUC-ROC: {auc_smote:.4f}")

print(f"\n📋 SMOTE Model Classification Report:")
print(classification_report(y_test, y_pred_smote))

print(f"🔲 SMOTE Model Confusion Matrix:")
cm_smote = confusion_matrix(y_test, y_pred_smote)
print(cm_smote)

# ==========================================
# STEP 5: COMPARE AND VISUALIZE 📈
# ==========================================

print("\n" + "="*70)
print("STEP 5: MODEL COMPARISON 📈")
print("="*70)

# Create comparison table
comparison_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Baseline': [accuracy_baseline, precision_baseline, recall_baseline, f1_baseline],
    'SMOTE': [accuracy_smote, precision_smote, recall_smote, f1_smote],
    'Improvement': [
        accuracy_smote - accuracy_baseline,
        precision_smote - precision_baseline,
        recall_smote - recall_baseline,
        f1_smote - f1_baseline
    ]
}

if auc_baseline and auc_smote:
    comparison_data['Metric'].append('AUC-ROC')
    comparison_data['Baseline'].append(auc_baseline)
    comparison_data['SMOTE'].append(auc_smote)
    comparison_data['Improvement'].append(auc_smote - auc_baseline)

comparison_df = pd.DataFrame(comparison_data)

print("\n📊 PERFORMANCE COMPARISON TABLE:")
print("-" * 60)
print(f"{'Metric':<15} {'Baseline':>12} {'SMOTE':>12} {'Improvement':>12}")
print("-" * 60)
for _, row in comparison_df.iterrows():
    print(f"{row['Metric']:<15} {row['Baseline']:>12.4f} {row['SMOTE']:>12.4f} {row['Improvement']:>+12.4f}")

# Per-class comparison for detailed analysis
print("\n📊 PER-CLASS PERFORMANCE COMPARISON:")
print("-" * 70)

# Get classification reports as dictionaries
report_baseline = classification_report(y_test, y_pred_baseline, output_dict=True)
report_smote = classification_report(y_test, y_pred_smote, output_dict=True)

# Compare per-class metrics
for class_label in np.unique(y_test):
    class_str = str(class_label)
    if class_str in report_baseline and class_str in report_smote:
        print(f"\nClass {class_label}:")
        print(f"   Precision: {report_baseline[class_str]['precision']:.4f} → {report_smote[class_str]['precision']:.4f} "
              f"({report_smote[class_str]['precision'] - report_baseline[class_str]['precision']:+.4f})")
        print(f"   Recall:    {report_baseline[class_str]['recall']:.4f} → {report_smote[class_str]['recall']:.4f} "
              f"({report_smote[class_str]['recall'] - report_baseline[class_str]['recall']:+.4f})")
        print(f"   F1-Score:  {report_baseline[class_str]['f1-score']:.4f} → {report_smote[class_str]['f1-score']:.4f} "
              f"({report_smote[class_str]['f1-score'] - report_baseline[class_str]['f1-score']:+.4f})")

# Identify minority class (assuming binary classification)
if len(np.unique(y)) == 2:
    minority_class = class_counts.idxmin()
    minority_class_str = str(minority_class)
    
    print(f"\n🎯 MINORITY CLASS ({minority_class}) IMPROVEMENT:")
    print(f"   Recall improvement: {report_smote[minority_class_str]['recall'] - report_baseline[minority_class_str]['recall']:+.4f}")
    print(f"   F1-Score improvement: {report_smote[minority_class_str]['f1-score'] - report_baseline[minority_class_str]['f1-score']:+.4f}")

# ==========================================
# VISUALIZATION
# ==========================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS 📊")
print("="*70)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Confusion Matrix - Baseline
ax1 = axes[0, 0]
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Baseline Model\nConfusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# 2. Confusion Matrix - SMOTE
ax2 = axes[0, 1]
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', ax=ax2)
ax2.set_title('SMOTE Model\nConfusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# 3. Metric Comparison Bar Chart
ax3 = axes[0, 2]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline_scores = [accuracy_baseline, precision_baseline, recall_baseline, f1_baseline]
smote_scores = [accuracy_smote, precision_smote, recall_smote, f1_smote]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, baseline_scores, width, label='Baseline', color='#3498db')
bars2 = ax3.bar(x + width/2, smote_scores, width, label='SMOTE', color='#2ecc71')

ax3.set_xlabel('Metrics')
ax3.set_ylabel('Score')
ax3.set_title('Model Performance Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.set_ylim([0, 1])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

# 4. ROC Curves (if binary classification)
ax4 = axes[1, 0]
if len(np.unique(y)) == 2:
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_proba_baseline[:, 1])
    fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred_proba_smote[:, 1])
    
    ax4.plot(fpr_baseline, tpr_baseline, color='#3498db', lw=2, 
             label=f'Baseline (AUC = {auc_baseline:.3f})')
    ax4.plot(fpr_smote, tpr_smote, color='#2ecc71', lw=2, 
             label=f'SMOTE (AUC = {auc_smote:.3f})')
    ax4.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves')
    ax4.legend(loc="lower right")
else:
    ax4.text(0.5, 0.5, 'ROC Curve\n(Binary Classification Only)', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_xticks([])
    ax4.set_yticks([])

# 5. Class Distribution Comparison
ax5 = axes[1, 1]
distribution_data = pd.DataFrame({
    'Original Train': y_train.value_counts().sort_index(),
    'SMOTE Train': pd.Series(y_resampled).value_counts().sort_index(),
    'Test Set': y_test.value_counts().sort_index()
})
distribution_data.plot(kind='bar', ax=ax5, color=['#e74c3c', '#2ecc71', '#3498db'])
ax5.set_title('Class Distribution Comparison')
ax5.set_xlabel('Class')
ax5.set_ylabel('Number of Samples')
ax5.legend(title='Dataset')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)

# 6. Improvement Summary
ax6 = axes[1, 2]
improvements = comparison_df['Improvement'].values[:4]  # First 4 metrics
colors_improvement = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
bars = ax6.bar(metrics, improvements, color=colors_improvement)
ax6.set_title('Performance Improvement\n(SMOTE vs Baseline)')
ax6.set_xlabel('Metrics')
ax6.set_ylabel('Improvement')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax6.set_ylim([min(improvements) - 0.05, max(improvements) + 0.05])

# Add value labels
for bar, val in zip(bars, improvements):
    ax6.text(bar.get_x() + bar.get_width()/2., val,
            f'{val:+.3f}', ha='center', 
            va='bottom' if val > 0 else 'top')

plt.suptitle('SMOTE Implementation Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save visualization
output_path = 'medisense/backend/data/final/optimized/'
plt.savefig(f'{output_path}smote_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved visualization: smote_analysis.png")

# ==========================================
# ADDITIONAL RESAMPLING STRATEGIES
# ==========================================

print("\n" + "="*70)
print("ADDITIONAL RESAMPLING STRATEGIES 🔬")
print("="*70)

# Try SMOTEENN (combines over and under sampling)
print("\n🔄 Testing SMOTEENN (SMOTE + Edited Nearest Neighbors)...")
smoteenn = SMOTEENN(random_state=42)
X_smoteenn, y_smoteenn = smoteenn.fit_resample(X_train_scaled, y_train)

print(f"   SMOTEENN samples: {X_smoteenn.shape[0]}")
print(f"   Class distribution: {pd.Series(y_smoteenn).value_counts().to_dict()}")

# Train model with SMOTEENN
rf_smoteenn = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smoteenn.fit(X_smoteenn, y_smoteenn)
y_pred_smoteenn = rf_smoteenn.predict(X_test_scaled)
f1_smoteenn = f1_score(y_test, y_pred_smoteenn, average='weighted')
print(f"   F1-Score: {f1_smoteenn:.4f}")

# ==========================================
# FINAL RECOMMENDATIONS
# ==========================================

print("\n" + "="*70)
print("FINAL RECOMMENDATIONS 💡")
print("="*70)

# Determine best approach
best_f1 = max(f1_baseline, f1_smote, f1_smoteenn)
if best_f1 == f1_baseline:
    best_method = "Baseline (with class weights)"
elif best_f1 == f1_smote:
    best_method = "SMOTE"
else:
    best_method = "SMOTEENN"

print(f"\n🏆 BEST METHOD: {best_method} (F1-Score: {best_f1:.4f})")

print(f"""
📊 KEY FINDINGS:
   1. Class Imbalance: {imbalance_ratio:.2f}:1 ratio in original data
   2. SMOTE Impact: {'Positive' if f1_smote > f1_baseline else 'Negative'} overall impact
   3. Minority Class Recall: {'Improved' if len(np.unique(y)) == 2 and report_smote[minority_class_str]['recall'] > report_baseline[minority_class_str]['recall'] else 'Check needed'}
   
🎯 RECOMMENDATIONS:
   1. {'Use SMOTE for this dataset' if f1_smote > f1_baseline else 'Consider alternative approaches'}
   2. Monitor minority class performance in production
   3. Consider cost-sensitive learning if misclassification costs differ
   4. Experiment with different SMOTE parameters (k_neighbors, sampling_strategy)
   5. Try ensemble methods combining multiple resampling strategies
   
⚠️  IMPORTANT REMINDERS:
   • ALWAYS apply SMOTE after train-test split
   • NEVER apply SMOTE to test data
   • Consider cross-validation for more robust evaluation
   • Monitor for overfitting on synthetic samples
""")

print("\n✅ SMOTE IMPLEMENTATION COMPLETE!")
print(f"   Models saved and ready for deployment")
print(f"   Remember: Use the {best_method} approach for production")

# Save the best model
import joblib
best_model = rf_smote if best_method == "SMOTE" else (rf_smoteenn if best_method == "SMOTEENN" else rf_baseline)
joblib.dump(best_model, f'{output_path}best_model_with_smote.pkl')
print(f"\n💾 Best model saved to: {output_path}best_model_with_smote.pkl")
