"""
Balanced Symptom Classifier Training
====================================
Train classifiers with proper class balancing techniques.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score, balanced_accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 80)
print("BALANCED SYMPTOM CLASSIFIER TRAINING")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n1. LOADING DATA...")
print("-" * 40)

# Load training and test sets
train_df = pd.read_csv('medisense/backend/data/final/symptom/symptom_train.csv')
test_df = pd.read_csv('medisense/backend/data/final/symptom/symptom_test.csv')

print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")

# Separate features and target
X_train = train_df.drop(['symptom_category_encoded', 'symptom_category_original'], axis=1)
y_train_encoded = train_df['symptom_category_encoded']
y_train_original = train_df['symptom_category_original']

X_test = test_df.drop(['symptom_category_encoded', 'symptom_category_original'], axis=1)
y_test_encoded = test_df['symptom_category_encoded']
y_test_original = test_df['symptom_category_original']

# Load the symptom encoder
le = joblib.load('medisense/backend/models/symptom_encoder.pkl')

print(f"\nClasses: {list(le.classes_)}")
print(f"Number of classes: {len(le.classes_)}")

# Check class distribution
print("\nTraining set class distribution:")
train_dist = y_train_original.value_counts()
for category, count in train_dist.items():
    percentage = (count / len(y_train_original)) * 100
    print(f"  {category:25s}: {count:5d} ({percentage:5.1f}%)")

# ============================================================================
# STEP 2: HANDLE MISSING VALUES
# ============================================================================

print("\n2. HANDLING MISSING VALUES...")
print("-" * 40)

# Fill NaN values
for col in X_train.columns:
    if X_train[col].isna().any():
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} with median: {median_val:.2f}")

# ============================================================================
# STEP 3: CLASS BALANCING WITH SMOTE
# ============================================================================

print("\n3. APPLYING CLASS BALANCING...")
print("-" * 40)

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weight_dict = dict(zip(np.unique(y_train_encoded), class_weights))

print("Class weights:")
for cls, weight in class_weight_dict.items():
    class_name = le.inverse_transform([cls])[0]
    print(f"  {class_name:25s}: {weight:.2f}")

# Apply SMOTE for oversampling
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)  # Use k_neighbors=3 for small classes
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_encoded)

print(f"Original training set: {len(X_train)} samples")
print(f"Balanced training set: {len(X_train_balanced)} samples")

# Check new distribution
balanced_dist = pd.Series(y_train_balanced).value_counts()
print("\nBalanced class distribution:")
for cls, count in balanced_dist.items():
    class_name = le.inverse_transform([cls])[0]
    percentage = (count / len(y_train_balanced)) * 100
    print(f"  {class_name:25s}: {count:5d} ({percentage:5.1f}%)")

# ============================================================================
# STEP 4: DEFINE MODELS WITH CLASS WEIGHTS
# ============================================================================

print("\n4. INITIALIZING MODELS...")
print("-" * 40)

models = {
    'Random Forest (Balanced)': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost (Weighted)': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        n_jobs=-1
    ),
    'Logistic Regression (Balanced)': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        solver='lbfgs'
    ),
    'Random Forest (SMOTE)': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

print(f"Models to train: {list(models.keys())}")

# ============================================================================
# STEP 5: TRAINING AND EVALUATION
# ============================================================================

print("\n5. TRAINING MODELS...")
print("-" * 40)

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use SMOTE data for the SMOTE model
    if 'SMOTE' in name:
        X_train_use = X_train_balanced
        y_train_use = y_train_balanced
    else:
        X_train_use = X_train
        y_train_use = y_train_encoded
    
    # Train model
    model.fit(X_train_use, y_train_use)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    balanced_acc = balanced_accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1,
        'f1_macro': f1_macro
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  F1-Score (weighted): {f1:.4f}")
    print(f"  F1-Score (macro): {f1_macro:.4f}")

# ============================================================================
# STEP 6: DETAILED EVALUATION OF BEST MODEL
# ============================================================================

print("\n6. DETAILED EVALUATION...")
print("-" * 40)

# Find best model based on balanced accuracy
best_model_name = max(results, key=lambda x: results[x]['balanced_accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name}")
print(f"Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.4f}")
print(f"F1-Score (macro): {results[best_model_name]['f1_macro']:.4f}")

# Classification report
print("\nClassification Report:")
print("-" * 40)
report = classification_report(y_test_encoded, best_predictions, 
                              target_names=le.classes_)
print(report)

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================

print("\n7. CREATING VISUALIZATIONS...")
print("-" * 40)

fig = plt.figure(figsize=(20, 10))

# 1. Model Comparison
ax1 = plt.subplot(2, 3, 1)
model_names = list(results.keys())
balanced_accs = [results[m]['balanced_accuracy'] for m in model_names]
f1_macros = [results[m]['f1_macro'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.bar(x - width/2, balanced_accs, width, label='Balanced Acc', color='steelblue')
bars2 = ax1.bar(x + width/2, f1_macros, width, label='F1 Macro', color='coral')

ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' ', '\n') for m in model_names], rotation=0, ha='center')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 2. Confusion Matrix for Best Model
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test_encoded, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax2)
ax2.set_title(f'Confusion Matrix - {best_model_name}')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# 3. Per-Class Performance
ax3 = plt.subplot(2, 3, 3)
report_dict = classification_report(y_test_encoded, best_predictions, 
                                   target_names=le.classes_, 
                                   output_dict=True)

classes = le.classes_
precision = [report_dict[c]['precision'] for c in classes]
recall = [report_dict[c]['recall'] for c in classes]
f1_class = [report_dict[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

ax3.bar(x - width, precision, width, label='Precision', color='lightblue')
ax3.bar(x, recall, width, label='Recall', color='lightgreen')
ax3.bar(x + width, f1_class, width, label='F1-Score', color='lightcoral')

ax3.set_xlabel('Symptom Category')
ax3.set_ylabel('Score')
ax3.set_title('Per-Class Performance')
ax3.set_xticks(x)
ax3.set_xticklabels([c[:10] for c in classes], rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Feature Importance (if available)
ax4 = plt.subplot(2, 3, 4)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    ax4.barh(range(15), importances[indices], color='steelblue')
    ax4.set_yticks(range(15))
    ax4.set_yticklabels([X_train.columns[i][:20] for i in indices])
    ax4.set_xlabel('Importance')
    ax4.set_title(f'Top 15 Features - {best_model_name}')
    ax4.grid(axis='x', alpha=0.3)

# 5. Class Distribution Comparison
ax5 = plt.subplot(2, 3, 5)
original_dist = y_train_original.value_counts()
balanced_dist_plot = pd.Series(le.inverse_transform(y_train_balanced)).value_counts()

x = np.arange(len(le.classes_))
width = 0.35

bars1 = ax5.bar(x - width/2, [original_dist.get(c, 0) for c in le.classes_], 
                width, label='Original', color='lightblue')
bars2 = ax5.bar(x + width/2, [balanced_dist_plot.get(c, 0) for c in le.classes_], 
                width, label='After SMOTE', color='lightcoral')

ax5.set_xlabel('Symptom Category')
ax5.set_ylabel('Count')
ax5.set_title('Training Data: Original vs Balanced')
ax5.set_xticks(x)
ax5.set_xticklabels([c[:10] for c in le.classes_], rotation=45, ha='right')
ax5.legend()

# 6. Performance Metrics Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
BEST MODEL: {best_model_name}

Performance Metrics:
• Accuracy: {results[best_model_name]['accuracy']:.3f}
• Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.3f}
• F1-Score (weighted): {results[best_model_name]['f1_weighted']:.3f}
• F1-Score (macro): {results[best_model_name]['f1_macro']:.3f}

Class Distribution (Test):
{test_df['symptom_category_original'].value_counts().to_string()}

Key Improvements:
✓ Class balancing with SMOTE
✓ Class weights for cost-sensitive learning
✓ Balanced accuracy metric
✓ Multiple model comparison
"""

ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace')
ax6.set_title('Summary', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('medisense/backend/data/visualization/final/balanced_classifier_evaluation.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 8: SAVE BEST MODEL
# ============================================================================

print("\n8. SAVING BEST MODEL...")
print("-" * 40)

# Save model
model_path = f'medisense/backend/models/balanced_symptom_classifier.pkl'
joblib.dump(best_model, model_path)
print(f"Best model saved to: {model_path}")

# Save results
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'best_model': best_model_name,
    'test_accuracy': float(results[best_model_name]['accuracy']),
    'balanced_accuracy': float(results[best_model_name]['balanced_accuracy']),
    'f1_weighted': float(results[best_model_name]['f1_weighted']),
    'f1_macro': float(results[best_model_name]['f1_macro']),
    'all_results': {k: {
        'accuracy': float(v['accuracy']),
        'balanced_accuracy': float(v['balanced_accuracy']),
        'f1_weighted': float(v['f1_weighted']),
        'f1_macro': float(v['f1_macro'])
    } for k, v in results.items()},
    'classes': list(le.classes_),
    'num_features': X_train.shape[1],
    'num_train_samples': len(X_train),
    'num_train_balanced': len(X_train_balanced),
    'num_test_samples': len(X_test)
}

with open('medisense/backend/data/final/symptom/balanced_classifier_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("Results saved to: medisense/backend/data/final/symptom/balanced_classifier_results.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
print(f"Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.4f}")
print(f"F1-Score (macro): {results[best_model_name]['f1_macro']:.4f}")

print("\nIMPROVEMENTS APPLIED:")
print("✓ Class balancing with SMOTE")
print("✓ Class weights for cost-sensitive learning")
print("✓ Balanced accuracy metric for imbalanced data")
print("✓ Multiple model architectures tested")

print("\n" + "=" * 80)
