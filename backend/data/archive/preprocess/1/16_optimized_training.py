"""
Optimized Model Training with Feature-Reduced Datasets
=======================================================
Uses the optimized datasets with proper feature handling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    mean_squared_error, r2_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("OPTIMIZED MODEL TRAINING WITH FEATURE-REDUCED DATASETS")
print("=" * 80)

def train_models(X_train, X_test, y_train, y_test, task_type, task_name):
    """Train Random Forest and XGBoost with proper parameters"""
    
    results = {}
    
    if task_type == 'regression':
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        }
        
        for name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            within_one = np.mean(np.abs(y_test - y_pred) <= 1) * 100
            
            results[name] = {
                'r2': r2,
                'rmse': rmse,
                'within_one': within_one
            }
            
            print(f"\n{name}:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Within ±1 visit: {within_one:.1f}%")
            
    else:  # Classification
        # Check class balance
        unique_classes = np.unique(y_train)
        class_counts = pd.Series(y_train).value_counts()
        min_class = class_counts.min()
        
        # Use SMOTE if imbalanced and enough samples
        use_smote = len(unique_classes) == 2 and min_class >= 6 and (class_counts.max() / min_class > 1.5)
        
        if use_smote:
            print(f"  Applying SMOTE (min class: {min_class})")
            k_neighbors = min(5, min_class - 1)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced = X_train
            y_train_balanced = y_train
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=10,
                class_weight='balanced' if not use_smote else None,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        for name, model in models.items():
            # Train
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            
            if len(unique_classes) == 2:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='binary', zero_division=0
                )
            else:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )
            
            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall: {rec:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # Show confusion matrix for classification
            if len(unique_classes) <= 10:
                cm = confusion_matrix(y_test, y_pred)
                print(f"  Confusion Matrix:\n{cm}")
    
    return results

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 80)
    print("LOADING OPTIMIZED DATASETS")
    print("=" * 80)
    
    all_results = {}
    
    # Task 1: Visit Count Regression
    print("\n📊 TASK 1: VISIT COUNT REGRESSION")
    print("-" * 60)
    
    try:
        # Load optimized dataset
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_regression_visits_optimized.csv')
        print(f"Loaded dataset: {df.shape}")
        
        # Prepare features and target
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        # CRITICAL: Remove current symptom counts for visit prediction
        symptom_cols = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
                       'dermatological_trauma_count', 'neuro_psych_count', 
                       'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count']
        
        cols_to_remove = [col for col in symptom_cols if col in X.columns]
        if cols_to_remove:
            print(f"Removing {len(cols_to_remove)} current symptom columns to prevent leakage")
            X = X.drop(columns=cols_to_remove)
        
        print(f"Final features: {X.shape[1]}")
        print(f"Zero-visit days: {(y == 0).mean()*100:.1f}%")
        
        # Split data (80-20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        results = train_models(X_train, X_test, y_train, y_test, 'regression', 'Visit Count')
        all_results['regression'] = results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Task 2: Binary Visit Classification
    print("\n📊 TASK 2: BINARY VISIT CLASSIFICATION")
    print("-" * 60)
    
    try:
        # Load optimized dataset
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_binary_visits_optimized.csv')
        print(f"Loaded dataset: {df.shape}")
        
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        # Remove current symptom counts
        symptom_cols = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
                       'dermatological_trauma_count', 'neuro_psych_count', 
                       'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count']
        
        cols_to_remove = [col for col in symptom_cols if col in X.columns]
        if cols_to_remove:
            print(f"Removing {len(cols_to_remove)} current symptom columns to prevent leakage")
            X = X.drop(columns=cols_to_remove)
        
        print(f"Final features: {X.shape[1]}")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        results = train_models(X_train, X_test, y_train, y_test, 'classification', 'Binary Visit')
        all_results['binary'] = results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Task 3: Dominant Symptom Classification
    print("\n📊 TASK 3: DOMINANT SYMPTOM CLASSIFICATION")
    print("-" * 60)
    
    try:
        # Load optimized dataset
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_dominant_symptom_optimized.csv')
        print(f"Loaded dataset: {df.shape}")
        
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        # For symptom classification, we KEEP the symptom counts
        print(f"Features: {X.shape[1]} (keeping symptom counts for this task)")
        print(f"Classes: {y.nunique()} unique symptoms")
        print(f"Distribution: {pd.Series(y).value_counts().head()}")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        results = train_models(X_train, X_test, y_train, y_test, 'classification', 'Dominant Symptom')
        all_results['dominant_symptom'] = results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    if 'regression' in all_results:
        print("\n🎯 Visit Count Regression:")
        best = max(all_results['regression'].items(), key=lambda x: -x[1]['rmse'])
        print(f"  Best Model: {best[0]}")
        print(f"  R² Score: {best[1]['r2']:.4f}")
        print(f"  RMSE: {best[1]['rmse']:.4f}")
        print(f"  Within ±1: {best[1]['within_one']:.1f}%")
    
    if 'binary' in all_results:
        print("\n🎯 Binary Visit Classification:")
        best = max(all_results['binary'].items(), key=lambda x: x[1]['f1'])
        print(f"  Best Model: {best[0]}")
        print(f"  Accuracy: {best[1]['accuracy']:.4f}")
        print(f"  F1-Score: {best[1]['f1']:.4f}")
    
    if 'dominant_symptom' in all_results:
        print("\n🎯 Dominant Symptom Classification:")
        best = max(all_results['dominant_symptom'].items(), key=lambda x: x[1]['f1'])
        print(f"  Best Model: {best[0]}")
        print(f"  Accuracy: {best[1]['accuracy']:.4f}")
        print(f"  F1-Score: {best[1]['f1']:.4f}")
    
    print("\n✅ Training complete with optimized datasets!")
    return all_results

if __name__ == "__main__":
    results = main()
