"""
Complete Final Model Training, Evaluation, and Saving
=====================================================
Trains models, saves them, generates metrics and visualizations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, r2_score, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('medisense/backend/models', exist_ok=True)
os.makedirs('medisense/backend/data/evaluation', exist_ok=True)
os.makedirs('medisense/backend/data/visualization/final', exist_ok=True)

print("=" * 80)
print("FINAL MODEL TRAINING AND EVALUATION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

def remove_current_symptoms(X):
    """Remove current symptom counts to prevent leakage"""
    symptom_cols = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
                   'dermatological_trauma_count', 'neuro_psych_count', 
                   'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count']
    
    cols_to_remove = [col for col in symptom_cols if col in X.columns]
    if cols_to_remove:
        print(f"   Removing {len(cols_to_remove)} current symptom columns")
        X = X.drop(columns=cols_to_remove)
    return X

def train_regression_models(X, y, task_name):
    """Train and evaluate regression models"""
    
    print(f"\n{'='*60}")
    print(f"Training Regression Models: {task_name}")
    print(f"{'='*60}")
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    }
    
    results = {}
    best_score = -float('inf')
    best_model = None
    best_model_name = None
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        fold_metrics = {'r2': [], 'rmse': [], 'within_1': []}
        all_y_true, all_y_pred = [], []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = np.maximum(model.predict(X_val), 0)
            
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            within_1 = np.mean(np.abs(y_val - y_pred) <= 1)
            
            fold_metrics['r2'].append(r2)
            fold_metrics['rmse'].append(rmse)
            fold_metrics['within_1'].append(within_1)
            
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            
            print(f"  Fold {fold}: R²={r2:.3f}, RMSE={rmse:.3f}, ±1={within_1:.1%}")
        
        avg_within_1 = np.mean(fold_metrics['within_1'])
        print(f"  Average: R²={np.mean(fold_metrics['r2']):.3f}, "
              f"RMSE={np.mean(fold_metrics['rmse']):.3f}, ±1={avg_within_1:.1%}")
        
        results[model_name] = {
            'metrics': {
                'r2': np.mean(fold_metrics['r2']),
                'rmse': np.mean(fold_metrics['rmse']),
                'within_1': avg_within_1
            },
            'fold_metrics': fold_metrics,
            'predictions': {'y_true': all_y_true, 'y_pred': all_y_pred}
        }
        
        if avg_within_1 > best_score:
            best_score = avg_within_1
            best_model = model
            best_model_name = model_name
    
    # Train final model on full dataset
    print(f"\n🏆 Best Model: {best_model_name} (±1: {best_score:.1%})")
    best_model.fit(X, y)
    
    # Save model
    model_path = f'medisense/backend/models/{task_name}_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        results[best_model_name]['feature_importance'] = feature_importance
    
    return results, best_model_name

def train_classification_models(X, y, task_name, is_multiclass=False):
    """Train and evaluate classification models"""
    
    print(f"\n{'='*60}")
    print(f"Training Classification Models: {task_name}")
    print(f"{'='*60}")
    
    class_dist = pd.Series(y).value_counts()
    print(f"Class distribution: {class_dist.to_dict()}")
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            class_weight='balanced', random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss'
        )
    }
    
    results = {}
    best_score = -float('inf')
    best_model = None
    best_model_name = None
    
    # Create label encoder for multiclass
    le = None
    if is_multiclass:
        le = LabelEncoder()
        le.fit(y)  # Fit on all labels to ensure consistency
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        fold_metrics = {'acc': [], 'f1': []}
        all_y_true, all_y_pred = [], []
        all_y_proba = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if len(np.unique(y_val)) < 2:
                print(f"  Fold {fold}: Skipped (only one class)")
                continue
            
            try:
                if model_name == 'XGBoost' and is_multiclass:
                    # Encode labels for XGBoost
                    y_train_encoded = le.transform(y_train)
                    y_val_encoded = le.transform(y_val)
                    model.fit(X_train, y_train_encoded)
                    y_pred_encoded = model.predict(X_val)
                    y_pred = le.inverse_transform(y_pred_encoded)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    if hasattr(model, 'predict_proba') and not is_multiclass:
                        y_proba = model.predict_proba(X_val)
                        all_y_proba.extend(y_proba[:, 1])
            except Exception as e:
                print(f"  Fold {fold}: Error - {str(e)}")
                continue
            
            acc = accuracy_score(y_val, y_pred)
            if is_multiclass:
                _, _, f1, _ = precision_recall_fscore_support(
                    y_val, y_pred, average='weighted', zero_division=0
                )
            else:
                _, _, f1, _ = precision_recall_fscore_support(
                    y_val, y_pred, average='binary', zero_division=0
                )
            
            fold_metrics['acc'].append(acc)
            fold_metrics['f1'].append(f1)
            
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            
            print(f"  Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}")
        
        if fold_metrics['f1']:  # If we have results
            avg_f1 = np.mean(fold_metrics['f1'])
            print(f"  Average: Acc={np.mean(fold_metrics['acc']):.3f}, F1={avg_f1:.3f}")
            
            results[model_name] = {
                'metrics': {
                    'accuracy': np.mean(fold_metrics['acc']),
                    'f1': avg_f1
                },
                'fold_metrics': fold_metrics,
                'predictions': {'y_true': all_y_true, 'y_pred': all_y_pred}
            }
            
            if all_y_proba:
                results[model_name]['predictions']['y_proba'] = all_y_proba
            
            if avg_f1 > best_score:
                best_score = avg_f1
                best_model = model
                best_model_name = model_name
    
    if best_model:
        # Train final model on full dataset
        print(f"\n🏆 Best Model: {best_model_name} (F1: {best_score:.3f})")
        
        if best_model_name == 'XGBoost' and is_multiclass:
            # Use the same label encoder that was already fitted
            y_encoded = le.transform(y)
            best_model.fit(X, y_encoded)
            joblib.dump(le, f'medisense/backend/models/{task_name}_label_encoder.pkl')
        else:
            best_model.fit(X, y)
        
        # Save model
        model_path = f'medisense/backend/models/{task_name}_model.pkl'
        joblib.dump(best_model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            results[best_model_name]['feature_importance'] = feature_importance
    
    return results, best_model_name

def create_visualizations(results, task_name, task_type='regression'):
    """Create evaluation visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{task_name} - Model Evaluation', fontsize=14, fontweight='bold')
    
    # Get best model
    best_model = max(results.items(), 
                    key=lambda x: x[1]['metrics'].get('within_1', x[1]['metrics'].get('f1', 0)))
    model_name, model_results = best_model
    
    if task_type == 'regression':
        # 1. Actual vs Predicted
        y_true = model_results['predictions']['y_true']
        y_pred = model_results['predictions']['y_pred']
        
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Visits')
        axes[0, 0].set_ylabel('Predicted Visits')
        axes[0, 0].set_title(f'Actual vs Predicted ({model_name})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = np.array(y_true) - np.array(y_pred)
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        
    else:  # Classification
        # 1. Confusion Matrix
        y_true = model_results['predictions']['y_true']
        y_pred = model_results['predictions']['y_pred']
        
        if task_type == 'binary':
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title(f'Confusion Matrix ({model_name})')
        else:
            # For multiclass, just show accuracy
            axes[0, 0].text(0.5, 0.5, f"Accuracy: {model_results['metrics']['accuracy']:.3f}\nF1: {model_results['metrics']['f1']:.3f}",
                           ha='center', va='center', fontsize=14)
            axes[0, 0].set_title(f'Performance ({model_name})')
        
        # 2. ROC Curve for binary
        if 'y_proba' in model_results['predictions']:
            y_proba = model_results['predictions']['y_proba']
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
    
    # 3. Model Comparison
    models = list(results.keys())
    if task_type == 'regression':
        metric_values = [results[m]['metrics']['within_1'] * 100 for m in models]
        axes[1, 0].bar(models, metric_values)
        axes[1, 0].set_ylabel('Within ±1 Visit (%)')
    else:
        metric_values = [results[m]['metrics']['f1'] for m in models]
        axes[1, 0].bar(models, metric_values)
        axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('Model Comparison')
    
    # 4. Feature Importance
    if 'feature_importance' in model_results:
        top_features = model_results['feature_importance'].head(10)
        axes[1, 1].barh(range(len(top_features)), top_features['importance'])
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features['feature'], fontsize=8)
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top 10 Features')
    
    plt.tight_layout()
    plt.savefig(f'medisense/backend/data/visualization/final/{task_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(all_results):
    """Save evaluation metrics to JSON"""
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj
    
    metrics_to_save = convert_to_serializable(all_results)
    
    with open('medisense/backend/data/evaluation/final_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print("\n✅ Metrics saved to: medisense/backend/data/evaluation/final_metrics.json")

def main():
    """Main training pipeline"""
    
    all_results = {}
    
    # Task 1: Visit Count Regression
    print("\n" + "="*80)
    print("TASK 1: VISIT COUNT REGRESSION")
    print("="*80)
    
    try:
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_regression_visits_optimized.csv')
        print(f"Dataset shape: {df.shape}")
        
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        X = remove_current_symptoms(X)
        
        print(f"Features: {X.shape[1]}, Zero-visit days: {(y == 0).mean()*100:.1f}%")
        
        results, best_model = train_regression_models(X, y, "visit_regression")
        all_results['visit_regression'] = {'results': results, 'best': best_model}
        create_visualizations(results, "Visit Regression", 'regression')
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Task 2: Binary Visit Classification
    print("\n" + "="*80)
    print("TASK 2: BINARY VISIT CLASSIFICATION")
    print("="*80)
    
    try:
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_binary_visits_optimized.csv')
        print(f"Dataset shape: {df.shape}")
        
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        X = remove_current_symptoms(X)
        
        print(f"Features: {X.shape[1]}")
        
        results, best_model = train_classification_models(X, y, "binary_visit", False)
        all_results['binary_visit'] = {'results': results, 'best': best_model}
        create_visualizations(results, "Binary Classification", 'binary')
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Task 3: Dominant Symptom Classification
    print("\n" + "="*80)
    print("TASK 3: DOMINANT SYMPTOM CLASSIFICATION")
    print("="*80)
    
    try:
        df = pd.read_csv('medisense/backend/data/final/optimized/dataset_dominant_symptom_optimized.csv')
        print(f"Dataset shape: {df.shape}")
        
        X = df.drop(columns=['target', 'date'], errors='ignore')
        y = df['target']
        
        print(f"Features: {X.shape[1]}, Classes: {y.nunique()}")
        
        results, best_model = train_classification_models(X, y, "dominant_symptom", True)
        all_results['dominant_symptom'] = {'results': results, 'best': best_model}
        create_visualizations(results, "Symptom Classification", 'multiclass')
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Save all metrics
    save_metrics(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for task, data in all_results.items():
        print(f"\n📊 {task.upper()}")
        print(f"  Best Model: {data['best']}")
        if 'results' in data and data['best'] in data['results']:
            metrics = data['results'][data['best']]['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\n📁 Outputs:")
    print("  - Models: medisense/backend/models/")
    print("  - Metrics: medisense/backend/data/evaluation/")
    print("  - Visualizations: medisense/backend/data/visualization/final/")

if __name__ == "__main__":
    main()
