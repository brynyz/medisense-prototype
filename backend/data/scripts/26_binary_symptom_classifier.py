"""
Binary Symptom Classifier
=========================
Simplify the problem to binary classification:
- Common symptoms (respiratory, pain_musculoskeletal, digestive) 
- vs Other symptoms
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score, balanced_accuracy_score,
                           roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 80)
print("BINARY SYMPTOM CLASSIFIER")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n1. LOADING AND PREPARING DATA...")
print("-" * 40)

# Load the full dataset
df = pd.read_csv('medisense/backend/data/final/symptom/individual_symptom_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Create binary target
# Common symptoms (>10% of data): respiratory, pain_musculoskeletal, digestive
common_symptoms = ['respiratory', 'pain_musculoskeletal', 'digestive']
df['is_common_symptom'] = df['symptom_category'].isin(common_symptoms).astype(int)

print("\nBinary Classification:")
print(f"  Common symptoms: {common_symptoms}")
print(f"  Common: {df['is_common_symptom'].sum()} ({df['is_common_symptom'].mean()*100:.1f}%)")
print(f"  Other: {(1-df['is_common_symptom']).sum()} ({(1-df['is_common_symptom']).mean()*100:.1f}%)")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print("\n2. FEATURE ENGINEERING...")
print("-" * 40)

# Create interaction features
df['temp_x_humidity'] = df['temp'] * df['humidity']
df['pm_total'] = df['pm2_5'] + df['pm10']
df['visit_trend'] = df['daily_visits_count'] - df['visit_lag1'].fillna(df['daily_visits_count'])
df['is_high_visit_day'] = (df['daily_visits_count'] > df['daily_visits_count'].median()).astype(int)

# Create binned features
df['temp_category'] = pd.cut(df['temp'], bins=3, labels=['low', 'medium', 'high'])
df['humidity_category'] = pd.cut(df['humidity'], bins=3, labels=['low', 'medium', 'high'])
df['age_bin'] = pd.cut(df['age'], bins=[0, 19, 21, 100], labels=['18-19', '20-21', '22+'])

# One-hot encode new categorical features
df = pd.get_dummies(df, columns=['temp_category', 'humidity_category', 'age_bin'], drop_first=True)

print(f"Features after engineering: {df.shape[1]}")

# ============================================================================
# STEP 3: PREPARE TRAIN/TEST SPLIT
# ============================================================================

print("\n3. SPLITTING DATA...")
print("-" * 40)

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['symptom_category', 'symptom_category_encoded', 'is_common_symptom']]
X = df[feature_cols]
y = df['is_common_symptom']

# Handle missing values
for col in X.columns:
    if X[col].isna().any():
        X[col].fillna(X[col].median(), inplace=True)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train distribution - Common: {y_train.mean():.2%}, Other: {(1-y_train).mean():.2%}")
print(f"Test distribution - Common: {y_test.mean():.2%}, Other: {(1-y_test).mean():.2%}")

# ============================================================================
# STEP 4: DEFINE MODELS
# ============================================================================

print("\n4. INITIALIZING MODELS...")
print("-" * 40)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(1-y_train).sum()/y_train.sum(),
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
}

print(f"Models to evaluate: {list(models.keys())}")

# ============================================================================
# STEP 5: CROSS-VALIDATION
# ============================================================================

print("\n5. PERFORMING CROSS-VALIDATION...")
print("-" * 40)

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
    
    cv_results[name] = {
        'accuracy_mean': cv_scores.mean(),
        'accuracy_std': cv_scores.std(),
        'f1_mean': cv_f1.mean(),
        'f1_std': cv_f1.std(),
        'roc_auc_mean': cv_roc.mean(),
        'roc_auc_std': cv_roc.std()
    }
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"  CV ROC-AUC: {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")

# ============================================================================
# STEP 6: TRAIN AND EVALUATE
# ============================================================================

print("\n6. TRAINING FINAL MODELS...")
print("-" * 40)

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# STEP 7: BEST MODEL EVALUATION
# ============================================================================

print("\n7. DETAILED EVALUATION...")
print("-" * 40)

# Find best model by ROC-AUC
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']
best_probabilities = results[best_model_name]['probabilities']

print(f"\nBest Model: {best_model_name}")
print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")

# Classification report
print("\nClassification Report:")
print("-" * 40)
report = classification_report(y_test, best_predictions, 
                              target_names=['Other', 'Common'])
print(report)

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    for idx, row in importances.head(15).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================

print("\n8. CREATING VISUALIZATIONS...")
print("-" * 40)

fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison
ax1 = plt.subplot(2, 4, 1)
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
roc_aucs = [results[m]['roc_auc'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
bars2 = ax1.bar(x + width/2, roc_aucs, width, label='ROC-AUC', color='coral')

ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' ', '\n') for m in model_names], rotation=0, ha='center', fontsize=8)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Confusion Matrix
ax2 = plt.subplot(2, 4, 2)
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Other', 'Common'], 
            yticklabels=['Other', 'Common'], ax=ax2)
ax2.set_title(f'Confusion Matrix - {best_model_name}')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# 3. ROC Curves
ax3 = plt.subplot(2, 4, 3)
for name in results:
    fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
    ax3.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax3.plot([0, 1], [0, 1], 'k--', label='Random')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves')
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(alpha=0.3)

# 4. Feature Importance (if available)
ax4 = plt.subplot(2, 4, 4)
if hasattr(best_model, 'feature_importances_'):
    importances_plot = importances.head(15)
    ax4.barh(range(15), importances_plot['importance'], color='steelblue')
    ax4.set_yticks(range(15))
    ax4.set_yticklabels([f[:20] for f in importances_plot['feature']], fontsize=8)
    ax4.set_xlabel('Importance')
    ax4.set_title(f'Top 15 Features - {best_model_name}')
    ax4.grid(axis='x', alpha=0.3)

# 5. Cross-validation scores
ax5 = plt.subplot(2, 4, 5)
cv_df = pd.DataFrame(cv_results).T
cv_df[['accuracy_mean', 'f1_mean', 'roc_auc_mean']].plot(kind='bar', ax=ax5)
ax5.set_xlabel('Model')
ax5.set_ylabel('CV Score')
ax5.set_title('Cross-Validation Performance')
ax5.legend(['Accuracy', 'F1-Score', 'ROC-AUC'])
ax5.grid(axis='y', alpha=0.3)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 6. Probability Distribution
ax6 = plt.subplot(2, 4, 6)
ax6.hist(best_probabilities[y_test == 0], bins=30, alpha=0.5, label='Other', color='blue')
ax6.hist(best_probabilities[y_test == 1], bins=30, alpha=0.5, label='Common', color='red')
ax6.set_xlabel('Predicted Probability')
ax6.set_ylabel('Frequency')
ax6.set_title(f'Probability Distribution - {best_model_name}')
ax6.legend()
ax6.grid(alpha=0.3)

# 7. Performance by threshold
ax7 = plt.subplot(2, 4, 7)
thresholds = np.linspace(0, 1, 100)
f1_scores = []
accuracies = []
for threshold in thresholds:
    y_pred_thresh = (best_probabilities >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))
    accuracies.append(accuracy_score(y_test, y_pred_thresh))

ax7.plot(thresholds, f1_scores, label='F1-Score', color='blue')
ax7.plot(thresholds, accuracies, label='Accuracy', color='red')
ax7.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax7.set_xlabel('Threshold')
ax7.set_ylabel('Score')
ax7.set_title('Performance by Threshold')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. Summary text
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')

summary_text = f"""
BINARY CLASSIFICATION RESULTS

Best Model: {best_model_name}

Performance Metrics:
• Accuracy: {results[best_model_name]['accuracy']:.3f}
• Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.3f}
• F1-Score: {results[best_model_name]['f1_score']:.3f}
• ROC-AUC: {results[best_model_name]['roc_auc']:.3f}

Classification Target:
• Common: respiratory, pain, digestive
• Other: remaining categories

Dataset:
• Total samples: {len(df)}
• Common: {df['is_common_symptom'].mean():.1%}
• Other: {(1-df['is_common_symptom']).mean():.1%}
"""

ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace')
ax8.set_title('Summary', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('medisense/backend/data/visualization/final/binary_symptom_classifier.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================

print("\n9. SAVING BEST MODEL...")
print("-" * 40)

# Save model
model_path = 'medisense/backend/models/binary_symptom_classifier.pkl'
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Save results
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'task': 'binary_symptom_classification',
    'best_model': best_model_name,
    'common_symptoms': common_symptoms,
    'metrics': {
        'accuracy': float(results[best_model_name]['accuracy']),
        'balanced_accuracy': float(results[best_model_name]['balanced_accuracy']),
        'f1_score': float(results[best_model_name]['f1_score']),
        'roc_auc': float(results[best_model_name]['roc_auc'])
    },
    'cv_results': cv_results,
    'all_model_results': {k: {
        'accuracy': float(v['accuracy']),
        'balanced_accuracy': float(v['balanced_accuracy']),
        'f1_score': float(v['f1_score']),
        'roc_auc': float(v['roc_auc'])
    } for k, v in results.items()},
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'common_percentage': float(df['is_common_symptom'].mean()),
        'features': list(X.columns)
    }
}

import json
with open('medisense/backend/data/final/symptom/binary_classifier_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("Results saved to: medisense/backend/data/final/symptom/binary_classifier_results.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("BINARY CLASSIFICATION COMPLETED")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")

print("\nThis binary classification approach simplifies the problem and")
print("should provide more reliable predictions for common vs uncommon symptoms.")

print("\n" + "=" * 80)
