"""
Dominant Symptom Classification - FIXED VERSION
================================================
This script properly handles XGBoost multiclass classification with string labels.

THE XGBOOST PROBLEM:
-------------------
XGBoost expects class labels to be consecutive integers starting from 0.
When some classes are missing in a fold (due to small dataset), XGBoost fails.

THE SOLUTION:
------------
1. Use LabelEncoder to map string labels to integers
2. Ensure all classes are known to the encoder from the start
3. Handle missing classes in validation folds gracefully

Author: MediSense Team
Date: September 2024
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DOMINANT SYMPTOM CLASSIFICATION (FIXED)")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

def load_and_analyze_data():
    """
    Load the dominant symptom dataset and analyze class distribution.
    """
    print("\n📁 Loading dominant symptom dataset...")
    
    df = pd.read_csv('medisense/backend/data/final/optimized/dataset_dominant_symptom_optimized.csv')
    print(f"Dataset shape: {df.shape}")
    
    X = df.drop(columns=['target', 'date'], errors='ignore')
    y = df['target']
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(y)}")
    print(f"Unique symptom classes: {y.nunique()}")
    
    # Analyze class distribution
    class_dist = pd.Series(y).value_counts()
    print("\n📊 Class Distribution:")
    print("-" * 40)
    for symptom, count in class_dist.items():
        percentage = (count / len(y)) * 100
        print(f"  {symptom:25s}: {count:3d} ({percentage:5.1f}%)")
    
    # Identify problematic classes
    print("\n⚠️ Class Imbalance Analysis:")
    if class_dist.min() < 5:
        print(f"  Severe imbalance detected!")
        print(f"  Smallest class: {class_dist.idxmin()} with {class_dist.min()} samples")
        print(f"  Largest class: {class_dist.idxmax()} with {class_dist.max()} samples")
        print(f"  Imbalance ratio: {class_dist.max() / class_dist.min():.1f}:1")
    
    return X, y, class_dist

def create_label_encoder(y):
    """
    Create and fit a label encoder for the symptom classes.
    
    IMPORTANT: This encoder maps string labels to consecutive integers.
    """
    le = LabelEncoder()
    le.fit(y)  # Fit on ALL labels to ensure consistency
    
    print("\n🔄 Label Encoding Mapping:")
    print("-" * 40)
    for i, label in enumerate(le.classes_):
        count = (y == label).sum()
        print(f"  {i} → {label:25s} ({count} samples)")
    
    return le

def train_random_forest(X, y):
    """
    Train Random Forest classifier (works directly with string labels).
    """
    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Check if we have enough classes
        unique_train = y_train.nunique()
        unique_val = y_val.nunique()
        
        print(f"\nFold {fold}:")
        print(f"  Training classes: {unique_train}, Validation classes: {unique_val}")
        
        if unique_val < 2:
            print(f"  ⚠️ Skipped: Only one class in validation")
            continue
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='weighted', zero_division=0
        )
        
        fold_scores.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})
        print(f"  Accuracy: {acc:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        # Show per-class performance for this fold
        unique_labels = sorted(set(y_val) | set(y_pred))
        print(f"  Classes in fold: {unique_labels}")
    
    if fold_scores:
        avg_acc = np.mean([s['acc'] for s in fold_scores])
        avg_f1 = np.mean([s['f1'] for s in fold_scores])
        print(f"\n📊 Random Forest Average: Acc={avg_acc:.3f}, F1={avg_f1:.3f}")
    else:
        avg_acc = avg_f1 = 0
        print("\n❌ Random Forest: No valid folds")
    
    return model, fold_scores, avg_acc, avg_f1

def train_xgboost_fixed(X, y, le):
    """
    Train XGBoost classifier with proper label encoding.
    
    KEY FIX: We encode ALL labels at the start and handle missing classes in folds.
    """
    print("\n" + "="*60)
    print("XGBOOST TRAINING (FIXED)")
    print("="*60)
    
    # Encode all labels
    y_encoded = le.transform(y)
    num_classes = len(le.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Encoded label range: {y_encoded.min()} to {y_encoded.max()}")
    
    # Configure XGBoost for multiclass
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,  # Higher value for small classes
        objective='multi:softprob',  # Multiclass with probability
        num_class=num_classes,  # CRITICAL: Specify number of classes
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False  # Disable XGBoost's internal label encoding
    )
    
    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_encoded = y_encoded[train_idx]
        y_val_encoded = y_encoded[val_idx]
        y_val_original = y.iloc[val_idx]
        
        # Check class distribution in this fold
        unique_train = len(np.unique(y_train_encoded))
        unique_val = len(np.unique(y_val_encoded))
        
        print(f"\nFold {fold}:")
        print(f"  Training classes: {unique_train}/{num_classes}")
        print(f"  Validation classes: {unique_val}/{num_classes}")
        print(f"  Validation class range: {y_val_encoded.min()}-{y_val_encoded.max()}")
        
        if unique_val < 2:
            print(f"  ⚠️ Skipped: Only one class in validation")
            continue
        
        try:
            # CRITICAL: Create sample weights for missing classes
            # This helps XGBoost understand the full class range
            sample_weights = np.ones(len(y_train_encoded))
            
            # Train model
            model.fit(
                X_train, 
                y_train_encoded,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val_encoded)],
                verbose=False
            )
            
            # Predict
            y_pred_encoded = model.predict(X_val)
            
            # Decode predictions back to original labels
            y_pred = le.inverse_transform(y_pred_encoded.astype(int))
            
            # Calculate metrics
            acc = accuracy_score(y_val_original, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val_original, y_pred, average='weighted', zero_division=0
            )
            
            fold_scores.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})
            print(f"  ✅ Success!")
            print(f"  Accuracy: {acc:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            # Try to understand the error
            if "Invalid classes" in str(e):
                print(f"  Debugging: Expected classes 0-{num_classes-1}")
                print(f"  Debugging: Got classes {np.unique(y_val_encoded)}")
    
    if fold_scores:
        avg_acc = np.mean([s['acc'] for s in fold_scores])
        avg_f1 = np.mean([s['f1'] for s in fold_scores])
        print(f"\n📊 XGBoost Average: Acc={avg_acc:.3f}, F1={avg_f1:.3f}")
    else:
        avg_acc = avg_f1 = 0
        print("\n❌ XGBoost: No valid folds")
    
    return model, fold_scores, avg_acc, avg_f1

def evaluate_final_model(model, X, y, model_name, le=None):
    """
    Evaluate the final model on the full dataset.
    """
    print("\n" + "="*60)
    print(f"FINAL EVALUATION: {model_name}")
    print("="*60)
    
    if model_name == "XGBoost" and le is not None:
        # For XGBoost, we need to encode inputs and decode outputs
        y_encoded = le.transform(y)
        y_pred_encoded = model.predict(X)
        y_pred = le.inverse_transform(y_pred_encoded.astype(int))
    else:
        # For Random Forest, direct prediction
        y_pred = model.predict(X)
    
    # Overall accuracy
    acc = accuracy_score(y, y_pred)
    print(f"Overall Accuracy: {acc:.3f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(y, y_pred, zero_division=0)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print("-" * 60)
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    return acc, report, cm, feature_importance if 'feature_importance' in locals() else None

def main():
    """
    Main execution: Train and evaluate symptom classification models.
    """
    import os
    os.makedirs('medisense/backend/models/symptom_classification', exist_ok=True)
    
    # Load and analyze data
    X, y, class_dist = load_and_analyze_data()
    
    # Create label encoder
    le = create_label_encoder(y)
    
    # Train Random Forest
    rf_model, rf_scores, rf_avg_acc, rf_avg_f1 = train_random_forest(X, y)
    
    # Train XGBoost (fixed version)
    xgb_model, xgb_scores, xgb_avg_acc, xgb_avg_f1 = train_xgboost_fixed(X, y, le)
    
    # Determine best model
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"Random Forest: Acc={rf_avg_acc:.3f}, F1={rf_avg_f1:.3f}")
    print(f"XGBoost:       Acc={xgb_avg_acc:.3f}, F1={xgb_avg_f1:.3f}")
    
    if rf_avg_f1 >= xgb_avg_f1:
        best_model = rf_model
        best_name = "RandomForest"
        best_f1 = rf_avg_f1
        use_le = False
    else:
        best_model = xgb_model
        best_name = "XGBoost"
        best_f1 = xgb_avg_f1
        use_le = True
    
    print(f"\n🏆 Best Model: {best_name} (F1={best_f1:.3f})")
    
    # Train best model on full dataset
    print(f"\nTraining final {best_name} model on full dataset...")
    if best_name == "XGBoost":
        y_encoded = le.transform(y)
        best_model.fit(X, y_encoded)
    else:
        best_model.fit(X, y)
    
    # Final evaluation
    final_acc, report, cm, feature_imp = evaluate_final_model(
        best_model, X, y, best_name, le if use_le else None
    )
    
    # Save models and results
    print("\n" + "="*60)
    print("SAVING MODELS AND RESULTS")
    print("="*60)
    
    # Save best model
    model_path = f'medisense/backend/models/symptom_classification/{best_name.lower()}_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save label encoder if using XGBoost
    if use_le:
        le_path = 'medisense/backend/models/symptom_classification/label_encoder.pkl'
        joblib.dump(le, le_path)
        print(f"✅ Label encoder saved: {le_path}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_name,
        'class_distribution': class_dist.to_dict(),
        'random_forest': {
            'avg_accuracy': float(rf_avg_acc),
            'avg_f1': float(rf_avg_f1),
            'fold_scores': rf_scores
        },
        'xgboost': {
            'avg_accuracy': float(xgb_avg_acc),
            'avg_f1': float(xgb_avg_f1),
            'fold_scores': xgb_scores
        },
        'final_accuracy': float(final_acc),
        'top_features': feature_imp.head(15).to_dict('records') if feature_imp is not None else None
    }
    
    with open('medisense/backend/models/symptom_classification/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved: results.json")
    
    print("\n" + "="*80)
    print("SYMPTOM CLASSIFICATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Performance: {final_acc:.1%} accuracy")
    
    return results

if __name__ == "__main__":
    main()