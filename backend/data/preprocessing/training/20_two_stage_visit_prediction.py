"""
Two-Stage Visit Prediction Model
=================================
This script implements a two-stage approach to handle zero-inflated medical visit data.

APPROACH:
---------
Stage 1: Binary Classification - Predict IF a visit will occur (0 or 1)
Stage 2: Regression - For predicted visit days, estimate HOW MANY visits

WHY TWO-STAGE?
--------------
With 85.8% zero-visit days, traditional regression struggles. By separating:
1. The "will there be visits?" question (binary)
2. The "how many visits?" question (regression)
We achieve better overall accuracy and interpretability.

Author: MediSense Team
Date: September 2024
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, r2_score, confusion_matrix
)
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TWO-STAGE VISIT PREDICTION MODEL")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

def load_and_prepare_data():
    """
    Load and prepare data for two-stage modeling.
    Returns separate datasets for binary and regression tasks.
    """
    print("\n📁 Loading datasets...")
    
    # Load binary classification dataset
    binary_df = pd.read_csv('medisense/backend/data/final/optimized/dataset_binary_visits_optimized.csv')
    print(f"Binary dataset shape: {binary_df.shape}")
    
    # Load regression dataset
    regression_df = pd.read_csv('medisense/backend/data/final/optimized/dataset_regression_visits_optimized.csv')
    print(f"Regression dataset shape: {regression_df.shape}")
    
    return binary_df, regression_df

def remove_current_symptoms(X):
    """
    Remove current-day symptom counts to prevent data leakage.
    
    IMPORTANT: Current symptom counts are perfect predictors of visits
    because symptoms only exist when visits occur. We must remove them
    and rely on lagged features instead.
    """
    symptom_cols = [
        'respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
        'dermatological_trauma_count', 'neuro_psych_count', 
        'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count'
    ]
    
    cols_to_remove = [col for col in symptom_cols if col in X.columns]
    if cols_to_remove:
        print(f"   Removing {len(cols_to_remove)} current symptom columns to prevent leakage")
        X = X.drop(columns=cols_to_remove)
    
    return X

def train_binary_stage(X, y):
    """
    Stage 1: Train binary classification models to predict visit/no-visit.
    
    This stage answers: "Will there be any visits today?"
    """
    print("\n" + "="*60)
    print("STAGE 1: BINARY CLASSIFICATION (Visit/No-Visit)")
    print("="*60)
    
    # Show class distribution
    class_dist = pd.Series(y).value_counts()
    print(f"Class distribution: No-visit={class_dist[0]}, Visit={class_dist[1]}")
    print(f"Imbalance ratio: {class_dist[0]/class_dist[1]:.2f}:1")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Define models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=class_dist[0]/class_dist[1],  # Handle imbalance
            random_state=42
        )
    }
    
    best_score = 0
    best_model = None
    best_model_name = None
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name} Training:")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary', zero_division=0
            )
            
            fold_scores.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})
            print(f"  Fold {fold}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
        
        # Calculate average scores
        avg_acc = np.mean([s['acc'] for s in fold_scores])
        avg_f1 = np.mean([s['f1'] for s in fold_scores])
        
        print(f"  Average: Acc={avg_acc:.3f}, F1={avg_f1:.3f}")
        
        results[model_name] = {
            'avg_acc': avg_acc,
            'avg_f1': avg_f1,
            'fold_scores': fold_scores
        }
        
        # Track best model
        if avg_f1 > best_score:
            best_score = avg_f1
            best_model = model
            best_model_name = model_name
    
    print(f"\n🏆 Best Binary Model: {best_model_name} (F1={best_score:.3f})")
    
    # Train final model on full dataset
    best_model.fit(X, y)
    
    return best_model, best_model_name, results

def train_regression_stage(X, y, binary_predictions=None):
    """
    Stage 2: Train regression models to predict visit count.
    
    This stage answers: "Given that visits will occur, how many?"
    
    If binary_predictions provided, only trains on predicted visit days.
    """
    print("\n" + "="*60)
    print("STAGE 2: REGRESSION (Visit Count)")
    print("="*60)
    
    # If we have binary predictions, filter to only predicted visit days
    if binary_predictions is not None:
        visit_mask = binary_predictions == 1
        X_filtered = X[visit_mask]
        y_filtered = y[visit_mask]
        print(f"Training on {visit_mask.sum()} predicted visit days")
    else:
        # For standalone regression, filter to actual visit days
        visit_mask = y > 0
        X_filtered = X[visit_mask]
        y_filtered = y[visit_mask]
        print(f"Training on {visit_mask.sum()} actual visit days")
    
    if len(X_filtered) < 50:
        print("⚠️ Warning: Very few visit days for regression training")
        return None, None, {}
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Define models
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,  # Lower for smaller dataset
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
    }
    
    best_score = float('inf')
    best_model = None
    best_model_name = None
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name} Training:")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_filtered), 1):
            if len(train_idx) < 10 or len(val_idx) < 5:
                print(f"  Fold {fold}: Skipped (insufficient data)")
                continue
            
            X_train, X_val = X_filtered.iloc[train_idx], X_filtered.iloc[val_idx]
            y_train, y_val = y_filtered.iloc[train_idx], y_filtered.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = np.mean(np.abs(y_val - y_pred))
            within_1 = np.mean(np.abs(y_val - y_pred) <= 1)
            
            fold_scores.append({'rmse': rmse, 'mae': mae, 'within_1': within_1})
            print(f"  Fold {fold}: RMSE={rmse:.3f}, MAE={mae:.3f}, ±1={within_1:.1%}")
        
        if fold_scores:
            # Calculate average scores
            avg_rmse = np.mean([s['rmse'] for s in fold_scores])
            avg_within_1 = np.mean([s['within_1'] for s in fold_scores])
            
            print(f"  Average: RMSE={avg_rmse:.3f}, ±1={avg_within_1:.1%}")
            
            results[model_name] = {
                'avg_rmse': avg_rmse,
                'avg_within_1': avg_within_1,
                'fold_scores': fold_scores
            }
            
            # Track best model (lower RMSE is better)
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_model = model
                best_model_name = model_name
    
    if best_model:
        print(f"\n🏆 Best Regression Model: {best_model_name} (RMSE={best_score:.3f})")
        
        # Train final model on all visit days
        best_model.fit(X_filtered, y_filtered)
    
    return best_model, best_model_name, results

def evaluate_two_stage_model(binary_model, regression_model, X, y):
    """
    Evaluate the complete two-stage model performance.
    
    This combines both stages to produce final predictions.
    """
    print("\n" + "="*60)
    print("TWO-STAGE MODEL EVALUATION")
    print("="*60)
    
    # Stage 1: Binary predictions
    binary_pred = binary_model.predict(X)
    binary_acc = accuracy_score(y > 0, binary_pred)
    
    print(f"Stage 1 (Binary) Accuracy: {binary_acc:.3f}")
    
    # Stage 2: Regression predictions (only for predicted visit days)
    final_predictions = np.zeros(len(X))
    visit_mask = binary_pred == 1
    
    if visit_mask.sum() > 0 and regression_model is not None:
        # Predict counts for predicted visit days
        final_predictions[visit_mask] = regression_model.predict(X[visit_mask])
        
        # Ensure predictions are non-negative integers
        final_predictions = np.maximum(final_predictions, 0)
        final_predictions = np.round(final_predictions)
    
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(y, final_predictions))
    mae = np.mean(np.abs(y - final_predictions))
    within_1 = np.mean(np.abs(y - final_predictions) <= 1)
    
    # Separate metrics for zero and non-zero days
    zero_mask = y == 0
    zero_acc = np.mean(final_predictions[zero_mask] == 0) if zero_mask.sum() > 0 else 0
    
    nonzero_mask = y > 0
    if nonzero_mask.sum() > 0:
        nonzero_recall = np.mean(final_predictions[nonzero_mask] > 0)
        nonzero_rmse = np.sqrt(mean_squared_error(y[nonzero_mask], final_predictions[nonzero_mask]))
    else:
        nonzero_recall = 0
        nonzero_rmse = 0
    
    print("\n📊 Overall Performance:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  Within ±1 visit: {within_1:.1%}")
    
    print("\n📊 Detailed Performance:")
    print(f"  Zero-day accuracy: {zero_acc:.1%}")
    print(f"  Non-zero day recall: {nonzero_recall:.1%}")
    print(f"  Non-zero day RMSE: {nonzero_rmse:.3f}")
    
    return {
        'overall': {
            'rmse': rmse,
            'mae': mae,
            'within_1': within_1
        },
        'zero_days': {
            'accuracy': zero_acc
        },
        'nonzero_days': {
            'recall': nonzero_recall,
            'rmse': nonzero_rmse
        }
    }

def main():
    """
    Main execution: Train and evaluate two-stage model.
    """
    import os
    os.makedirs('medisense/backend/models/two_stage', exist_ok=True)
    
    # Load data
    binary_df, regression_df = load_and_prepare_data()
    
    # Prepare binary classification data
    print("\n" + "-"*60)
    print("Preparing Binary Classification Data")
    print("-"*60)
    X_binary = binary_df.drop(columns=['target', 'date'], errors='ignore')
    y_binary = binary_df['target']
    X_binary = remove_current_symptoms(X_binary)
    print(f"Binary features: {X_binary.shape[1]}")
    
    # Train Stage 1: Binary Classification
    binary_model, binary_name, binary_results = train_binary_stage(X_binary, y_binary)
    
    # Prepare regression data
    print("\n" + "-"*60)
    print("Preparing Regression Data")
    print("-"*60)
    X_regression = regression_df.drop(columns=['target', 'date'], errors='ignore')
    y_regression = regression_df['target']
    X_regression = remove_current_symptoms(X_regression)
    print(f"Regression features: {X_regression.shape[1]}")
    
    # Get binary predictions for regression training
    # Note: In practice, we'd use the same features, but here we align datasets
    binary_pred_for_regression = binary_model.predict(X_regression)
    
    # Train Stage 2: Regression
    regression_model, regression_name, regression_results = train_regression_stage(
        X_regression, y_regression, binary_pred_for_regression
    )
    
    # Evaluate complete two-stage model
    evaluation_results = evaluate_two_stage_model(
        binary_model, regression_model, X_regression, y_regression
    )
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    joblib.dump(binary_model, f'medisense/backend/models/two_stage/binary_{binary_name}.pkl')
    print(f"✅ Binary model saved: binary_{binary_name}.pkl")
    
    if regression_model:
        joblib.dump(regression_model, f'medisense/backend/models/two_stage/regression_{regression_name}.pkl')
        print(f"✅ Regression model saved: regression_{regression_name}.pkl")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'binary_stage': {
            'best_model': binary_name,
            'results': binary_results
        },
        'regression_stage': {
            'best_model': regression_name,
            'results': regression_results
        },
        'two_stage_evaluation': evaluation_results
    }
    
    with open('medisense/backend/models/two_stage/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("✅ Results saved: results.json")
    
    print("\n" + "="*80)
    print("TWO-STAGE MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Performance: {evaluation_results['overall']['within_1']:.1%} within ±1 visit")
    
    return results

if __name__ == "__main__":
    main()
