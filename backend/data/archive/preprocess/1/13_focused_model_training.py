"""
Focused Model Training: Regression, Binary Visit, and Dominant Symptom
=======================================================================
Simplified training focusing on three key tasks with Random Forest and XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, r2_score, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FOCUSED MODEL TRAINING: KEY TASKS ONLY")
print("=" * 80)

def train_and_evaluate(X, y, task_type, task_name):
    """Train and evaluate Random Forest and XGBoost models"""
    
    print(f"\n{'='*80}")
    print(f"Task: {task_name}")
    print(f"Dataset shape: {X.shape}")
    print(f"Target type: {task_type}")
    print("-" * 80)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove date column if present
    if 'date' in X.columns:
        X = X.drop(columns=['date'])
    
    # Split data (80-20 split, maintaining temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    results = {}
    
    if task_type == 'regression':
        # Regression models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        print("\n📈 Training Regression Models:")
        
        for name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Within ±1 visit accuracy
            within_one_train = np.mean(np.abs(y_train - y_pred_train) <= 1) * 100
            within_one_test = np.mean(np.abs(y_test - y_pred_test) <= 1) * 100
            
            print(f"\n{name}:")
            print(f"  Training   -> R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, ±1 visit: {within_one_train:.1f}%")
            print(f"  Testing    -> R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, ±1 visit: {within_one_test:.1f}%")
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'within_one_test': within_one_test,
                'model': model
            }
            
    else:  # Classification
        # Check class distribution
        print(f"\nClass distribution in training set:")
        train_dist = pd.Series(y_train).value_counts()
        for cls, count in train_dist.items():
            print(f"  {cls}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Determine if we need SMOTE
        min_class = train_dist.min()
        max_class = train_dist.max()
        imbalance_ratio = max_class / min_class
        
        use_smote = imbalance_ratio > 1.5 and min_class >= 6
        
        if use_smote:
            print(f"\n⚖️  Using SMOTE (imbalance ratio: {imbalance_ratio:.2f}:1)")
            k_neighbors = min(5, min_class - 1)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"  After SMOTE: {len(X_train_balanced)} samples")
        else:
            print(f"\n✓ Classes are balanced enough (ratio: {imbalance_ratio:.2f}:1)")
            X_train_balanced = X_train
            y_train_balanced = y_train
        
        # Classification models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced' if not use_smote else None,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        print("\n🎯 Training Classification Models:")
        
        for name, model in models.items():
            # Train
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            # Get F1 scores
            if len(np.unique(y_train)) > 2:
                # Multi-class
                train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
                    y_train, y_pred_train, average='weighted', zero_division=0
                )
                test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
                    y_test, y_pred_test, average='weighted', zero_division=0
                )
            else:
                # Binary
                train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
                    y_train, y_pred_train, average='binary', zero_division=0
                )
                test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
                    y_test, y_pred_test, average='binary', zero_division=0
                )
            
            print(f"\n{name}:")
            print(f"  Training   -> Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Testing    -> Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
            print(f"              Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
            
            # Confusion matrix for test set
            cm = confusion_matrix(y_test, y_pred_test)
            print(f"  Confusion Matrix:")
            print(f"  {cm}")
            
            results[name] = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'model': model
            }
            
            # Feature importance (top 10)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"\n  Top 10 Important Features:")
                for idx, row in feature_importance.head(10).iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return results

def main():
    """Main training function for three key tasks"""
    
    all_results = {}
    
    # Task 1: Visit Count Regression
    print("\n" + "="*80)
    print("TASK 1: VISIT COUNT REGRESSION")
    print("="*80)
    
    try:
        # Try optimized dataset first, fallback to regular
        try:
            df = pd.read_csv('medisense/backend/data/final/optimized/dataset_regression_visits_optimized.csv')
            print("📂 Using optimized (feature-reduced) dataset")
        except:
            df = pd.read_csv('medisense/backend/data/final/dataset_regression_visits.csv')
            print("📂 Using regular dataset")
            
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        # Remove current symptom counts to avoid leakage (if they exist)
        symptom_current = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
                          'dermatological_trauma_count', 'neuro_psych_count', 
                          'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count']
        X = X.drop(columns=[col for col in symptom_current if col in X.columns])
        
        results = train_and_evaluate(X, y, 'regression', 'Visit Count Prediction')
        all_results['regression'] = results
        
    except Exception as e:
        print(f"❌ Error in regression task: {str(e)}")
    
    # Task 2: Binary Visit Classification
    print("\n" + "="*80)
    print("TASK 2: BINARY VISIT CLASSIFICATION")
    print("="*80)
    
    try:
        # Try optimized dataset first, fallback to regular
        try:
            df = pd.read_csv('medisense/backend/data/final/optimized/dataset_binary_visits_optimized.csv')
            print("📂 Using optimized (feature-reduced) dataset")
        except:
            df = pd.read_csv('medisense/backend/data/final/dataset_binary_visits.csv')
            print("📂 Using regular dataset")
            
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        # Remove current symptom counts to avoid leakage (if they exist)
        symptom_current = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
                          'dermatological_trauma_count', 'neuro_psych_count', 
                          'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count']
        X = X.drop(columns=[col for col in symptom_current if col in X.columns])
        
        results = train_and_evaluate(X, y, 'classification', 'Binary Visit Prediction')
        all_results['binary_visit'] = results
        
    except Exception as e:
        print(f"❌ Error in binary classification task: {str(e)}")
    
    # Task 3: Dominant Symptom Classification
    print("\n" + "="*80)
    print("TASK 3: DOMINANT SYMPTOM CLASSIFICATION")
    print("="*80)
    
    try:
        # Try optimized dataset first, fallback to regular
        try:
            df = pd.read_csv('medisense/backend/data/final/optimized/dataset_dominant_symptom_optimized.csv')
            print("📂 Using optimized (feature-reduced) dataset")
        except:
            df = pd.read_csv('medisense/backend/data/final/dataset_dominant_symptom.csv')
            print("📂 Using regular dataset")
            
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        # For symptom classification, we can keep current symptom counts
        # since we're predicting which symptom is dominant
        
        results = train_and_evaluate(X, y, 'classification', 'Dominant Symptom Prediction')
        all_results['dominant_symptom'] = results
        
    except Exception as e:
        print(f"❌ Error in dominant symptom task: {str(e)}")
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\n📊 BEST MODELS:")
    
    # Regression summary
    if 'regression' in all_results:
        print("\n1. Visit Count Regression:")
        best_model = max(all_results['regression'].items(), 
                        key=lambda x: x[1]['test_r2'] if 'test_r2' in x[1] else -999)
        print(f"   Best: {best_model[0]}")
        print(f"   Test R²: {best_model[1]['test_r2']:.4f}")
        print(f"   Test RMSE: {best_model[1]['test_rmse']:.4f}")
        print(f"   Within ±1 visit: {best_model[1]['within_one_test']:.1f}%")
    
    # Binary classification summary
    if 'binary_visit' in all_results:
        print("\n2. Binary Visit Classification:")
        best_model = max(all_results['binary_visit'].items(), 
                        key=lambda x: x[1]['test_f1'] if 'test_f1' in x[1] else -999)
        print(f"   Best: {best_model[0]}")
        print(f"   Test Accuracy: {best_model[1]['test_acc']:.4f}")
        print(f"   Test F1-Score: {best_model[1]['test_f1']:.4f}")
        print(f"   Test Precision: {best_model[1]['test_precision']:.4f}")
        print(f"   Test Recall: {best_model[1]['test_recall']:.4f}")
    
    # Dominant symptom summary
    if 'dominant_symptom' in all_results:
        print("\n3. Dominant Symptom Classification:")
        best_model = max(all_results['dominant_symptom'].items(), 
                        key=lambda x: x[1]['test_f1'] if 'test_f1' in x[1] else -999)
        print(f"   Best: {best_model[0]}")
        print(f"   Test Accuracy: {best_model[1]['test_acc']:.4f}")
        print(f"   Test F1-Score: {best_model[1]['test_f1']:.4f}")
        print(f"   Test Precision: {best_model[1]['test_precision']:.4f}")
        print(f"   Test Recall: {best_model[1]['test_recall']:.4f}")
    
    print("\n✅ Training complete!")
    return all_results

if __name__ == "__main__":
    results = main()
