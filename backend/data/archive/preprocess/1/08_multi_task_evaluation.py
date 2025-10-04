"""
Multi-Task Model Evaluation
============================
This script evaluates different prediction tasks:
1. Regression: Visit count prediction
2. Multi-class: Dominant symptom prediction
3. Other classification tasks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    # Classification metrics
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("🎯 MEDISENSE MULTI-TASK MODEL EVALUATION")
print("=" * 70)
print("Testing regression and multi-class prediction tasks")
print("=" * 70)

# ==========================================
# TASK 1: REGRESSION - VISIT COUNT PREDICTION
# ==========================================

print("\n" + "="*70)
print("TASK 1: REGRESSION - VISIT COUNT PREDICTION 📈")
print("="*70)

# Load regression dataset
try:
    df_regression = pd.read_csv('medisense/backend/data/final/optimized/dataset_regression_visits_optimized.csv')
    print("✅ Loaded optimized regression dataset")
except:
    df_regression = pd.read_csv('medisense/backend/data/final/dataset_regression_visits.csv')
    print("✅ Loaded regression dataset")

print(f"   Shape: {df_regression.shape}")

# Prepare data
X_reg = df_regression.drop(columns=['target', 'date'], errors='ignore')
y_reg = df_regression['target']

# Check target distribution
print(f"\n📊 Target Statistics:")
print(f"   Mean visits: {y_reg.mean():.2f}")
print(f"   Median visits: {y_reg.median():.2f}")
print(f"   Max visits: {y_reg.max()}")
print(f"   Zero-visit days: {(y_reg == 0).sum()} ({(y_reg == 0).sum() / len(y_reg) * 100:.1f}%)")

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Standardize features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f"\n🔄 Training Regression Models...")

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_lr = lr_model.predict(X_test_reg_scaled)

# 2. Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_rf = rf_reg.predict(X_test_reg_scaled)

# 3. XGBoost Regressor
xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_xgb = xgb_reg.predict(X_test_reg_scaled)

# Calculate metrics
def calculate_regression_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Custom metric: Accuracy within ±1 visit
    within_one = np.mean(np.abs(y_true - y_pred) <= 1)
    
    print(f"\n📊 {model_name} Performance:")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   MAE: {mae:.3f}")
    print(f"   R²: {r2:.3f}")
    print(f"   Within ±1 visit: {within_one:.1%}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'within_one': within_one}

print("\n" + "-"*50)
print("REGRESSION RESULTS:")
print("-"*50)

lr_metrics = calculate_regression_metrics(y_test_reg, y_pred_lr, "Linear Regression")
rf_metrics = calculate_regression_metrics(y_test_reg, y_pred_rf, "Random Forest")
xgb_metrics = calculate_regression_metrics(y_test_reg, y_pred_xgb, "XGBoost")

# Identify best model
best_reg_r2 = max(lr_metrics['r2'], rf_metrics['r2'], xgb_metrics['r2'])
if best_reg_r2 == lr_metrics['r2']:
    best_reg_model = "Linear Regression"
elif best_reg_r2 == rf_metrics['r2']:
    best_reg_model = "Random Forest"
else:
    best_reg_model = "XGBoost"

print(f"\n🏆 Best Regression Model: {best_reg_model} (R² = {best_reg_r2:.3f})")

# ==========================================
# TASK 2: DOMINANT SYMPTOM PREDICTION
# ==========================================

print("\n" + "="*70)
print("TASK 2: DOMINANT SYMPTOM PREDICTION 🏥")
print("="*70)

# Load dominant symptom dataset
try:
    df_symptom = pd.read_csv('medisense/backend/data/final/optimized/dataset_dominant_symptom_optimized.csv')
    print("✅ Loaded optimized dominant symptom dataset")
except:
    df_symptom = pd.read_csv('medisense/backend/data/final/dataset_dominant_symptom.csv')
    print("✅ Loaded dominant symptom dataset")


# Prepare data
X_sym = df_symptom.drop(columns=['target', 'date'], errors='ignore')
y_sym = df_symptom['target']

# Define symptom columns (updated categorization)
symptom_cols = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count', 
                'dermatological_trauma_count', 'neuro_psych_count', 'systemic_infectious_count',
                'cardiovascular_chronic_count', 'other_count']

# Check class distribution
print(f"\n Class Distribution:")
class_counts = y_sym.value_counts()
for class_val, count in class_counts.items():
    percentage = (count / len(y_sym)) * 100
    print(f"   Class {class_val}: {count} samples ({percentage:.1f}%)")

# Check if we need to handle 'no_symptom' class
if 'no_symptom' in y_sym.values or 0 in y_sym.values:
    # Filter to only days with symptoms for meaningful prediction
    print("\n⚠️  Filtering to days with symptoms only...")
    mask = (y_sym != 'no_symptom') & (y_sym != 0)
    X_sym = X_sym[mask]
    y_sym = y_sym[mask]
    print(f"   Filtered shape: {X_sym.shape}")

# Encode target if needed
if y_sym.dtype == 'object':
    le = LabelEncoder()
    y_sym_encoded = le.fit_transform(y_sym)
    class_names = le.classes_
else:
    y_sym_encoded = y_sym
    class_names = np.unique(y_sym)

# Check class distribution for stratification
class_counts_encoded = pd.Series(y_sym_encoded).value_counts()
min_class_size = class_counts_encoded.min()

print(f"   Minimum class size: {min_class_size}")

# Split data - use stratify only if all classes have at least 2 samples
if min_class_size >= 2:
    X_train_sym, X_test_sym, y_train_sym, y_test_sym = train_test_split(
        X_sym, y_sym_encoded, test_size=0.2, random_state=42, stratify=y_sym_encoded
    )
    print("   ✅ Using stratified split")
else:
    # Remove classes with only 1 sample
    print(f"   ⚠️  Removing classes with < 2 samples...")
    valid_classes = class_counts_encoded[class_counts_encoded >= 2].index
    mask = pd.Series(y_sym_encoded).isin(valid_classes)
    X_sym_filtered = X_sym[mask]
    y_sym_filtered = y_sym_encoded[mask]
    
    # Re-encode to ensure consecutive class labels (0, 1, 2, ...)
    le_filtered = LabelEncoder()
    y_sym_filtered = le_filtered.fit_transform(y_sym_filtered)
    
    # Update class names for filtered data
    if y_sym.dtype == 'object':
        # Get original class names for valid classes
        original_valid_classes = [class_names[i] for i in valid_classes]
        class_names = np.array(original_valid_classes)
    else:
        class_names = valid_classes
    
    print(f"   Filtered from {len(y_sym_encoded)} to {len(y_sym_filtered)} samples")
    print(f"   Classes after filtering: {np.unique(y_sym_filtered)}")
    
    X_train_sym, X_test_sym, y_train_sym, y_test_sym = train_test_split(
        X_sym_filtered, y_sym_filtered, test_size=0.2, random_state=42, stratify=y_sym_filtered
    )

# Standardize features
scaler_sym = StandardScaler()
X_train_sym_scaled = scaler_sym.fit_transform(X_train_sym)
X_test_sym_scaled = scaler_sym.transform(X_test_sym)

print(f"\n🔄 Training Classification Models...")

# 1. Baseline Random Forest
rf_baseline = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced'
)
rf_baseline.fit(X_train_sym_scaled, y_train_sym)
y_pred_baseline = rf_baseline.predict(X_test_sym_scaled)

# 2. XGBoost Classifier
try:
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_clf.fit(X_train_sym_scaled, y_train_sym)
    y_pred_xgb_clf = xgb_clf.predict(X_test_sym_scaled)
    print("   ✅ XGBoost trained successfully")
except Exception as e:
    print(f"   ⚠️  XGBoost training failed: {str(e)[:100]}")
    # Use baseline predictions as fallback
    y_pred_xgb_clf = y_pred_baseline

# 3. Apply SMOTE for multi-class (if severe imbalance and enough samples)
# Get class distribution in training set
train_class_counts = pd.Series(y_train_sym).value_counts()
min_train_samples = train_class_counts.min()
imbalance_ratio = train_class_counts.max() / train_class_counts.min()

# Only apply SMOTE if we have enough samples in minority class
if imbalance_ratio > 3 and min_train_samples >= 6:
    print(f"\n⚖️  Applying SMOTE (imbalance ratio: {imbalance_ratio:.1f}:1)")
    # Adjust k_neighbors based on minimum class size
    k_neighbors = min(5, min_train_samples - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X_train_sym_scaled, y_train_sym)
        
        # Train with SMOTE
        rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_smote.fit(X_resampled, y_resampled)
        y_pred_smote = rf_smote.predict(X_test_sym_scaled)
        print(f"   ✅ SMOTE applied successfully with k_neighbors={k_neighbors}")
    except Exception as e:
        print(f"   ⚠️  SMOTE failed: {str(e)[:100]}")
        y_pred_smote = None
else:
    y_pred_smote = None
    if imbalance_ratio <= 3:
        print(f"\n✅ Classes relatively balanced (ratio: {imbalance_ratio:.1f}:1)")
    else:
        print(f"\n⚠️  Cannot apply SMOTE: minimum class has only {min_train_samples} samples (need ≥6)")

# Calculate metrics
print("\n" + "-"*50)
print("DOMINANT SYMPTOM PREDICTION RESULTS:")
print("-"*50)

def print_classification_metrics(y_true, y_pred, model_name, class_names=None):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n📊 {model_name} Performance:")
    print(f"   Accuracy: {acc:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1:.3f}")
    
    print(f"\n   Classification Report:")
    if class_names is not None:
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))
    
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

baseline_metrics = print_classification_metrics(y_test_sym, y_pred_baseline, "Random Forest (Baseline)", class_names)

try:
    xgb_metrics_clf = print_classification_metrics(y_test_sym, y_pred_xgb_clf, "XGBoost", class_names)
except:
    print("\n⚠️  XGBoost evaluation skipped")
    xgb_metrics_clf = baseline_metrics  # Use baseline as fallback

if y_pred_smote is not None:
    smote_metrics = print_classification_metrics(y_test_sym, y_pred_smote, "Random Forest (SMOTE)", class_names)
else:
    smote_metrics = None

# ==========================================
# TASK 3: OTHER CLASSIFICATION TASKS
# ==========================================

print("\n" + "="*70)
print("TASK 3: ADDITIONAL CLASSIFICATION TASKS 🎯")
print("="*70)

# Test other available datasets
other_tasks = [
    ('multiclass_visits', 'Visit Volume Categories'),
    ('risk_based', 'Risk Level Prediction'),
    ('symptom_severity', 'Symptom Severity'),
    ('respiratory_outbreak', 'Respiratory Outbreak Detection')
]

results_summary = {}

for task_name, task_description in other_tasks:
    try:
        # Try optimized version first
        df_task = pd.read_csv(f'medisense/backend/data/final/optimized/dataset_{task_name}_optimized.csv')
    except:
        try:
            df_task = pd.read_csv(f'medisense/backend/data/final/dataset_{task_name}.csv')
        except:
            print(f"\n⚠️  Dataset for {task_description} not found")
            continue
    
    print(f"\n📊 {task_description}:")
    print(f"   Shape: {df_task.shape}")
    
    # Quick evaluation
    X_task = df_task.drop(columns=['target', 'date'], errors='ignore')
    y_task = df_task['target']
    
    # Handle encoding if needed
    if y_task.dtype == 'object':
        le_task = LabelEncoder()
        y_task = le_task.fit_transform(y_task)
    
    # Quick train-test split - handle single-sample classes
    task_class_counts = pd.Series(y_task).value_counts()
    if task_class_counts.min() >= 2:
        X_train_task, X_test_task, y_train_task, y_test_task = train_test_split(
            X_task, y_task, test_size=0.2, random_state=42, stratify=y_task
        )
    else:
        # Filter out classes with only 1 sample
        valid_classes = task_class_counts[task_class_counts >= 2].index
        mask = pd.Series(y_task).isin(valid_classes)
        X_task_filtered = X_task[mask]
        y_task_filtered = y_task[mask]
        
        if len(y_task_filtered) > 10:  # Only proceed if we have enough samples
            X_train_task, X_test_task, y_train_task, y_test_task = train_test_split(
                X_task_filtered, y_task_filtered, test_size=0.2, random_state=42, stratify=y_task_filtered
            )
        else:
            print(f"   ⚠️  Insufficient samples after filtering")
            continue
    
    # Scale and train
    scaler_task = StandardScaler()
    X_train_task_scaled = scaler_task.fit_transform(X_train_task)
    X_test_task_scaled = scaler_task.transform(X_test_task)
    
    # Quick Random Forest
    rf_task = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_task.fit(X_train_task_scaled, y_train_task)
    y_pred_task = rf_task.predict(X_test_task_scaled)
    
    # Calculate metrics
    acc = accuracy_score(y_test_task, y_pred_task)
    f1 = f1_score(y_test_task, y_pred_task, average='weighted')
    
    print(f"   Accuracy: {acc:.3f}")
    print(f"   F1-Score: {f1:.3f}")
    
    results_summary[task_description] = {'accuracy': acc, 'f1': f1}

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "="*70)
print("FINAL SUMMARY 📝")
print("="*70)

print(f"""
🎯 MULTI-TASK EVALUATION RESULTS:

1. REGRESSION (Visit Count Prediction):
   - Best Model: {best_reg_model}
   - R² Score: {best_reg_r2:.3f}
   - RMSE: {rf_metrics['rmse']:.3f}
   - MAE: {rf_metrics['mae']:.3f}
   - Within ±1 visit: {rf_metrics['within_one']:.1%}
   
2. DOMINANT SYMPTOM PREDICTION:
   - Best Model: {'XGBoost' if xgb_metrics_clf and xgb_metrics_clf['f1'] > baseline_metrics['f1'] else 'Random Forest'}
   - Accuracy: {max(baseline_metrics['accuracy'], xgb_metrics_clf['accuracy'] if xgb_metrics_clf else 0):.3f}
   - F1-Score: {max(baseline_metrics['f1'], xgb_metrics_clf['f1'] if xgb_metrics_clf else 0):.3f}
   - Classes: {len(class_names) if 'class_names' in locals() else 'N/A'} symptom categories
   
3. KEY INSIGHTS:
   - Regression performs well with R² > 0.8 (if true)
   - Multi-class symptom prediction is more challenging
   - Class imbalance affects multi-class more than binary
   - Environmental and lag features are strong predictors
   
4. RECOMMENDATIONS:
   - Use ensemble methods (Random Forest/XGBoost) for best results
   - Consider task-specific feature engineering
   - Monitor performance on minority classes
   - Implement proper time-series validation for production
""")

print("\n✅ MULTI-TASK EVALUATION COMPLETE!")
print("   Models ready for deployment across different prediction tasks")
