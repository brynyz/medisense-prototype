"""
Symptom Category Classifier Training
====================================
Train and evaluate multiple classifiers for individual symptom category prediction.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 80)
print("SYMPTOM CATEGORY CLASSIFIER TRAINING")
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

# Separate features and target (already encoded)
X_train = train_df.drop(['symptom_category_encoded', 'symptom_category_original'], axis=1)
y_train_encoded = train_df['symptom_category_encoded']
y_train_original = train_df['symptom_category_original']
X_test = test_df.drop(['symptom_category_encoded', 'symptom_category_original'], axis=1)
y_test_encoded = test_df['symptom_category_encoded']
y_test_original = test_df['symptom_category_original']

# Load the symptom encoder to get class names
le = joblib.load('medisense/backend/models/symptom_encoder.pkl')

print(f"\nClasses: {list(le.classes_)}")
print(f"Number of classes: {len(le.classes_)}")

# Check for NaN values and handle them
print("\nChecking for NaN values...")
if X_train.isna().any().any():
    print("Warning: Training features contain NaN values!")
    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    print(f"Columns with NaN: {nan_cols}")
    # Fill NaN values with median for numerical columns
    for col in nan_cols:
        X_train[col].fillna(X_train[col].median(), inplace=True)
        X_test[col].fillna(X_test[col].median(), inplace=True)
    print("NaN values filled with median")
else:
    print("No NaN values found in training features")

if X_test.isna().any().any():
    print("Warning: Test features contain NaN values after filling!")
    print(X_test.isna().sum()[X_test.isna().sum() > 0])

# ============================================================================
# STEP 2: DEFINE MODELS
# ============================================================================

print("\n2. INITIALIZING MODELS...")
print("-" * 40)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        multi_class='multinomial'
    ),
    'Naive Bayes': GaussianNB()
}

print(f"Models to train: {list(models.keys())}")

{{ ... }}
# ============================================================================
# STEP 3: CROSS-VALIDATION
# ============================================================================

print("\n3. PERFORMING CROSS-VALIDATION...")
print("-" * 40)

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train_encoded, 
                               cv=skf, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(model, X_train, y_train_encoded, 
                               cv=skf, scoring='f1_weighted', n_jobs=-1)
    
    cv_results[name] = {
        'accuracy_mean': cv_scores.mean(),
        'accuracy_std': cv_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std()
    }
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  CV F1-Score: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")

# ============================================================================
# STEP 4: TRAIN FINAL MODELS
# ============================================================================

print("\n4. TRAINING FINAL MODELS...")
print("-" * 40)

trained_models = {}
test_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train_encoded)
    trained_models[name] = model
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    
    test_results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test F1-Score: {f1:.4f}")

# ============================================================================
# STEP 5: DETAILED EVALUATION OF BEST MODEL
# ============================================================================

print("\n5. DETAILED EVALUATION...")
print("-" * 40)

# Find best model based on F1-score
best_model_name = max(test_results, key=lambda x: test_results[x]['f1_score'])
best_model = trained_models[best_model_name]
best_predictions = test_results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {test_results[best_model_name]['accuracy']:.4f}")
print(f"Test F1-Score: {test_results[best_model_name]['f1_score']:.4f}")

# Classification report
print("\nClassification Report:")
print("-" * 40)
report = classification_report(y_test_encoded, best_predictions, 
                              target_names=le.classes_)
print(report)

# Per-class metrics
report_dict = classification_report(y_test_encoded, best_predictions, 
                                   target_names=le.classes_, 
                                   output_dict=True)

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n6. CREATING VISUALIZATIONS...")
print("-" * 40)

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison
ax1 = plt.subplot(2, 3, 1)
model_names = list(test_results.keys())
accuracies = [test_results[m]['accuracy'] for m in model_names]
f1_scores = [test_results[m]['f1_score'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1-Score', color='coral')

ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

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
ax3.set_xticklabels(classes, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Feature Importance (if available)
ax4 = plt.subplot(2, 3, 4)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    ax4.barh(range(15), importances[indices], color='steelblue')
    ax4.set_yticks(range(15))
    ax4.set_yticklabels([X_train.columns[i] for i in indices])
    ax4.set_xlabel('Importance')
    ax4.set_title(f'Top 15 Features - {best_model_name}')
    ax4.grid(axis='x', alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Feature importance not available', 
            ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Feature Importance')

# 5. Class Distribution in Test Set
ax5 = plt.subplot(2, 3, 5)
test_dist = pd.Series(y_test_original).value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(test_dist)))
ax5.pie(test_dist.values, labels=test_dist.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax5.set_title('Test Set Class Distribution')

# 6. Cross-Validation Scores
ax6 = plt.subplot(2, 3, 6)
cv_data = pd.DataFrame(cv_results).T
cv_data[['accuracy_mean', 'f1_mean']].plot(kind='bar', ax=ax6, 
                                           color=['steelblue', 'coral'])
ax6.set_xlabel('Model')
ax6.set_ylabel('CV Score')
ax6.set_title('Cross-Validation Performance')
ax6.legend(['CV Accuracy', 'CV F1-Score'])
ax6.grid(axis='y', alpha=0.3)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('medisense/backend/data/visualization/final/symptom_classifier_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 7: SAVE MODELS AND RESULTS
# ============================================================================

print("\n7. SAVING MODELS AND RESULTS...")
print("-" * 40)

# Save best model
model_path = f'medisense/backend/models/symptom_classifier_{best_model_name.lower().replace(" ", "_")}.pkl'
joblib.dump(best_model, model_path)
print(f"Best model saved to: {model_path}")

# Save results summary
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'best_model': best_model_name,
    'test_accuracy': float(test_results[best_model_name]['accuracy']),
    'test_f1_score': float(test_results[best_model_name]['f1_score']),
    'cv_results': cv_results,
    'test_results': {k: {'accuracy': float(v['accuracy']), 
                         'f1_score': float(v['f1_score'])} 
                     for k, v in test_results.items()},
    'classes': list(le.classes_),
    'num_features': X_train.shape[1],
    'num_train_samples': len(X_train),
    'num_test_samples': len(X_test)
}

with open('medisense/backend/data/final/symptom/symptom_classifier_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("Results saved to: medisense/backend/data/final/symptom/symptom_classifier_results.json")

# ============================================================================
# STEP 8: FINAL SUMMARY
# ============================================================================
print("TRAINING COMPLETED")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {test_results[best_model_name]['accuracy']:.4f}")
print(f"Test F1-Score: {test_results[best_model_name]['f1_score']:.4f}")

print("\nTop Performing Classes:")
for cls in le.classes_:
    if report_dict[cls]['f1-score'] > 0.7:
        print(f"  {cls}: F1={report_dict[cls]['f1-score']:.3f}")

print("\nChallenging Classes:")
for cls in le.classes_:
    if report_dict[cls]['f1-score'] < 0.5:
        print(f"  {cls}: F1={report_dict[cls]['f1-score']:.3f}")

print("\n" + "=" * 80)
print("Next steps:")
print("1. Review visualizations in symptom_classifier_evaluation.png")
print("2. Analyze per-class performance for insights")
print("3. Consider ensemble methods if needed")
print("4. Deploy model using saved files")
print("=" * 80)
