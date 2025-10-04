"""
Improved Focused Model Training with Better Handling of Imbalanced Data
========================================================================
Addresses zero-inflation and distribution shift issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, r2_score, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("IMPROVED FOCUSED MODEL TRAINING")
print("=" * 80)

def evaluate_with_time_series_cv(X, y, task_type, task_name):
    """Use time series cross-validation instead of single split"""
    
    print(f"\n{'='*80}")
    print(f"Task: {task_name}")
    print(f"Dataset shape: {X.shape}")
    print("-" * 80)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove date column if present
    if 'date' in X.columns:
        X = X.drop(columns=['date'])
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    results = {}
    
    if task_type == 'regression':
        # For regression with zero-inflation, try different approaches
        print("\n📈 Regression with Zero-Inflation Handling:")
        
        # Approach 1: Standard regression
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=8,  # Reduced depth to prevent overfitting
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,  # Reduced depth
                learning_rate=0.05,  # Lower learning rate
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        }
        
        for name, model in models.items():
            cv_scores = {'r2': [], 'rmse': [], 'within_one': []}
            
            print(f"\n{name} - Time Series CV:")
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Clip predictions to be non-negative
                y_pred = np.maximum(y_pred, 0)
                
                # Evaluate
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                within_one = np.mean(np.abs(y_test - y_pred) <= 1) * 100
                
                cv_scores['r2'].append(r2)
                cv_scores['rmse'].append(rmse)
                cv_scores['within_one'].append(within_one)
                
                print(f"  Fold {fold}: R²={r2:.3f}, RMSE={rmse:.3f}, ±1={within_one:.1f}%")
            
            # Average scores
            avg_r2 = np.mean(cv_scores['r2'])
            avg_rmse = np.mean(cv_scores['rmse'])
            avg_within = np.mean(cv_scores['within_one'])
            
            print(f"  Average: R²={avg_r2:.3f}, RMSE={avg_rmse:.3f}, ±1={avg_within:.1f}%")
            
            results[name] = {
                'avg_r2': avg_r2,
                'avg_rmse': avg_rmse,
                'avg_within_one': avg_within,
                'cv_scores': cv_scores
            }
        
        # Approach 2: Two-stage model (classification + regression)
        print("\n🎯 Two-Stage Approach (Binary + Regression):")
        
        # Stage 1: Binary classification (visit/no-visit)
        y_binary = (y > 0).astype(int)
        
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )
        
        binary_scores = []
        regression_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_binary_train = y_binary.iloc[train_idx]
            y_binary_test = y_binary.iloc[test_idx]
            
            # Stage 1: Predict if there will be visits
            clf.fit(X_train, y_binary_train)
            y_binary_pred = clf.predict(X_test)
            binary_acc = accuracy_score(y_binary_test, y_binary_pred)
            binary_scores.append(binary_acc)
            
            # Stage 2: For predicted visits, predict count
            if y_train[y_train > 0].shape[0] > 10:  # Need enough samples
                reg = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                
                # Train only on non-zero visits
                X_train_visits = X_train[y_train > 0]
                y_train_visits = y_train[y_train > 0]
                
                if len(X_train_visits) > 0:
                    reg.fit(X_train_visits, y_train_visits)
                    
                    # Predict
                    y_pred_final = np.zeros(len(y_test))
                    visit_mask = y_binary_pred == 1
                    
                    if visit_mask.sum() > 0:
                        y_pred_final[visit_mask] = reg.predict(X_test[visit_mask])
                        y_pred_final = np.maximum(y_pred_final, 0)
                    
                    # Evaluate
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
                    within_one = np.mean(np.abs(y_test - y_pred_final) <= 1) * 100
                    regression_scores.append({'rmse': rmse, 'within_one': within_one})
        
        if binary_scores and regression_scores:
            print(f"  Binary Stage Avg Accuracy: {np.mean(binary_scores):.3f}")
            print(f"  Overall Avg RMSE: {np.mean([s['rmse'] for s in regression_scores]):.3f}")
            print(f"  Overall Avg ±1: {np.mean([s['within_one'] for s in regression_scores]):.1f}%")
            
            results['Two-Stage'] = {
                'binary_acc': np.mean(binary_scores),
                'avg_rmse': np.mean([s['rmse'] for s in regression_scores]),
                'avg_within_one': np.mean([s['within_one'] for s in regression_scores])
            }
    
    else:  # Classification
        print(f"\n🎯 Classification Task:")
        
        # Check class distribution
        class_dist = y.value_counts()
        print(f"Class distribution: {class_dist.to_dict()}")
        
        # Determine if we need SMOTE
        min_class = class_dist.min()
        imbalance_ratio = class_dist.max() / min_class
        use_smote = imbalance_ratio > 1.5 and min_class >= 6
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        for name, model in models.items():
            cv_scores = {'acc': [], 'f1': [], 'precision': [], 'recall': []}
            
            print(f"\n{name} - Time Series CV:")
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Apply SMOTE if needed
                if use_smote and len(np.unique(y_train)) > 1:
                    try:
                        min_class_train = pd.Series(y_train).value_counts().min()
                        k_neighbors = min(5, min_class_train - 1)
                        if k_neighbors > 0:
                            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                            X_train, y_train = smote.fit_resample(X_train, y_train)
                    except:
                        pass  # Skip SMOTE if it fails
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Evaluate
                acc = accuracy_score(y_test, y_pred)
                
                # Calculate F1 with appropriate average
                avg_type = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average=avg_type, zero_division=0
                )
                
                cv_scores['acc'].append(acc)
                cv_scores['f1'].append(f1)
                cv_scores['precision'].append(prec)
                cv_scores['recall'].append(rec)
                
                print(f"  Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}")
            
            # Average scores
            avg_acc = np.mean(cv_scores['acc'])
            avg_f1 = np.mean(cv_scores['f1'])
            avg_prec = np.mean(cv_scores['precision'])
            avg_rec = np.mean(cv_scores['recall'])
            
            print(f"  Average: Acc={avg_acc:.3f}, F1={avg_f1:.3f}, Prec={avg_prec:.3f}, Rec={avg_rec:.3f}")
            
            results[name] = {
                'avg_acc': avg_acc,
                'avg_f1': avg_f1,
                'avg_precision': avg_prec,
                'avg_recall': avg_rec,
                'cv_scores': cv_scores
            }
    
    return results

def main():
    """Main training function"""
    
    all_results = {}
    
    # Task 1: Visit Count Regression
    print("\n" + "="*80)
    print("TASK 1: VISIT COUNT REGRESSION")
    print("="*80)
    
    try:
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_regression_visits_optimized.csv')
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        # Remove current symptom counts
        symptom_current = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
                          'dermatological_trauma_count', 'neuro_psych_count', 
                          'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count']
        X = X.drop(columns=[col for col in symptom_current if col in X.columns])
        
        print(f"Zero-visit days: {(y == 0).mean()*100:.1f}%")
        
        results = evaluate_with_time_series_cv(X, y, 'regression', 'Visit Count Prediction')
        all_results['regression'] = results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Task 2: Binary Visit Classification
    print("\n" + "="*80)
    print("TASK 2: BINARY VISIT CLASSIFICATION")
    print("="*80)
    
    try:
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_binary_visits_optimized.csv')
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        # Remove current symptom counts
        symptom_current = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
                          'dermatological_trauma_count', 'neuro_psych_count', 
                          'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count']
        X = X.drop(columns=[col for col in symptom_current if col in X.columns])
        
        results = evaluate_with_time_series_cv(X, y, 'classification', 'Binary Visit Prediction')
        all_results['binary_visit'] = results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Task 3: Dominant Symptom Classification
    print("\n" + "="*80)
    print("TASK 3: DOMINANT SYMPTOM CLASSIFICATION")
    print("="*80)
    
    try:
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_dominant_symptom_optimized.csv')
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        print(f"Number of symptom classes: {y.nunique()}")
        
        results = evaluate_with_time_series_cv(X, y, 'classification', 'Dominant Symptom Prediction')
        all_results['dominant_symptom'] = results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\n📊 BEST PERFORMING MODELS:")
    
    if 'regression' in all_results:
        print("\n1. Visit Count Regression:")
        # Find best by RMSE (lower is better)
        best_model = min([(k, v) for k, v in all_results['regression'].items() if 'avg_rmse' in v],
                        key=lambda x: x[1]['avg_rmse'])
        print(f"   Best: {best_model[0]}")
        if 'avg_r2' in best_model[1]:
            print(f"   Avg R²: {best_model[1].get('avg_r2', 'N/A'):.3f}")
        print(f"   Avg RMSE: {best_model[1]['avg_rmse']:.3f}")
        print(f"   Avg ±1 visit: {best_model[1]['avg_within_one']:.1f}%")
    
    if 'binary_visit' in all_results:
        print("\n2. Binary Visit Classification:")
        best_model = max(all_results['binary_visit'].items(), 
                        key=lambda x: x[1]['avg_f1'])
        print(f"   Best: {best_model[0]}")
        print(f"   Avg Accuracy: {best_model[1]['avg_acc']:.3f}")
        print(f"   Avg F1-Score: {best_model[1]['avg_f1']:.3f}")
    
    if 'dominant_symptom' in all_results:
        print("\n3. Dominant Symptom Classification:")
        best_model = max(all_results['dominant_symptom'].items(), 
                        key=lambda x: x[1]['avg_f1'])
        print(f"   Best: {best_model[0]}")
        print(f"   Avg Accuracy: {best_model[1]['avg_acc']:.3f}")
        print(f"   Avg F1-Score: {best_model[1]['avg_f1']:.3f}")
    
    print("\n✅ Training complete!")
    return all_results

if __name__ == "__main__":
    results = main()
