"""
Comprehensive Model Training with All Symptom Categories
=========================================================
This script trains models on all datasets with proper handling of:
- All 8 symptom categories
- Lag features for each symptom type
- SMOTE for class imbalance
- Time series validation
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, r2_score, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPREHENSIVE MODEL TRAINING WITH ALL SYMPTOM CATEGORIES")
print("=" * 80)

# Define all symptom categories
SYMPTOM_CATEGORIES = [
    'respiratory', 'digestive', 'pain_musculoskeletal', 
    'dermatological_trauma', 'neuro_psych', 'systemic_infectious',
    'cardiovascular_chronic', 'other'
]

def identify_symptom_features(feature_names):
    """Identify which features are related to symptoms"""
    symptom_features = {
        'current': [],
        'lag': [],
        'rolling': []
    }
    
    for feature in feature_names:
        # Check if it's a symptom-related feature
        for symptom in SYMPTOM_CATEGORIES:
            if symptom in feature:
                if 'lag' in feature:
                    symptom_features['lag'].append(feature)
                elif 'rolling' in feature:
                    symptom_features['rolling'].append(feature)
                elif f'{symptom}_count' == feature:
                    symptom_features['current'].append(feature)
                break
    
    return symptom_features

def remove_leaky_features(X, task_type):
    """Remove features that could cause data leakage based on task type"""
    features_to_remove = []
    
    if task_type == 'visit_prediction':
        # Remove current symptom counts for visit prediction
        symptom_current = [f'{s}_count' for s in SYMPTOM_CATEGORIES]
        features_to_remove.extend([f for f in symptom_current if f in X.columns])
        
        # Remove features that are only non-zero when visits occur
        leaky_features = ['total_symptom_load', 'symptom_diversity', 
                         'respiratory_dominance', 'fever_respiratory_combo']
        features_to_remove.extend([f for f in leaky_features if f in X.columns])
    
    # Remove features that don't exist
    features_to_remove = [f for f in features_to_remove if f in X.columns]
    
    if features_to_remove:
        print(f"   Removing {len(features_to_remove)} potentially leaky features")
        X_clean = X.drop(columns=features_to_remove)
    else:
        X_clean = X.copy()
    
    return X_clean

def train_model_with_smote(X, y, model_type='classification', task_name=''):
    """Train model with SMOTE for handling class imbalance"""
    
    print(f"\n📊 Training: {task_name}")
    print("-" * 60)
    
    # Identify symptom features
    symptom_features = identify_symptom_features(X.columns)
    print(f"   Symptom features found:")
    print(f"     - Current counts: {len(symptom_features['current'])}")
    print(f"     - Lag features: {len(symptom_features['lag'])}")
    print(f"     - Rolling features: {len(symptom_features['rolling'])}")
    
    # Remove leaky features based on task
    if 'visit' in task_name.lower() and 'symptom' not in task_name.lower():
        X_clean = remove_leaky_features(X, 'visit_prediction')
    else:
        X_clean = X.copy()
    
    print(f"   Final feature count: {X_clean.shape[1]}")
    
    # Handle missing values
    X_clean = X_clean.fillna(X_clean.median())
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = {}
    
    if model_type == 'classification':
        # Check class distribution
        class_dist = y.value_counts()
        print(f"   Class distribution: {class_dist.to_dict()}")
        
        # Determine if we need SMOTE
        min_class_size = class_dist.min()
        max_class_size = class_dist.max()
        imbalance_ratio = max_class_size / min_class_size
        
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Only use SMOTE if there's significant imbalance and enough samples
        use_smote = imbalance_ratio > 1.5 and min_class_size >= 5
        
        if use_smote:
            print(f"   Using SMOTE to balance classes")
            
            # Determine k_neighbors for SMOTE
            k_neighbors = min(5, min_class_size - 1)
            
            # Create models with SMOTE
            models = {
                'Random Forest + SMOTE': ImbPipeline([
                    ('smote', SMOTE(k_neighbors=k_neighbors, random_state=42)),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                ]),
                'XGBoost + SMOTEENN': ImbPipeline([
                    ('smoteenn', SMOTEENN(smote=SMOTE(k_neighbors=k_neighbors, random_state=42))),
                    ('classifier', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
                ]),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
        else:
            print(f"   Not using SMOTE (balanced or insufficient samples)")
            
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
        
        # Train and evaluate models
        best_score = 0
        best_model = None
        best_model_name = None
        
        for name, model in models.items():
            try:
                # Cross-validation
                scores = cross_val_score(model, X_clean, y, cv=tscv, scoring='f1_weighted')
                mean_score = scores.mean()
                
                print(f"   {name}: F1={mean_score:.4f} (+/- {scores.std():.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"   {name}: Failed - {str(e)}")
        
        if best_model:
            # Train best model on full data
            best_model.fit(X_clean, y)
            
            # Get predictions for final evaluation
            y_pred = best_model.predict(X_clean)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            
            # Handle multi-class metrics
            if len(class_dist) > 2:
                # Multi-class
                precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
                try:
                    # Get probability predictions for AUC
                    if hasattr(best_model, 'predict_proba'):
                        y_proba = best_model.predict_proba(X_clean)
                        # For multi-class, we can't compute single AUC easily
                        auc = None
                    else:
                        auc = None
                except:
                    auc = None
            else:
                # Binary classification
                precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
                try:
                    if hasattr(best_model, 'predict_proba'):
                        y_proba = best_model.predict_proba(X_clean)[:, 1]
                        auc = roc_auc_score(y, y_proba)
                    else:
                        auc = None
                except:
                    auc = None
            
            results = {
                'model': best_model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'best_model': best_model
            }
            
            print(f"\n   ✅ Best Model: {best_model_name}")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall: {recall:.4f}")
            print(f"      F1-Score: {f1:.4f}")
            if auc:
                print(f"      AUC-ROC: {auc:.4f}")
            
            # Feature importance for tree-based models
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
            elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                importances = best_model.named_steps['classifier'].feature_importances_
            else:
                importances = None
            
            if importances is not None:
                # Get top features
                feature_importance = pd.DataFrame({
                    'feature': X_clean.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"\n   📈 Top 10 Important Features:")
                for idx, row in feature_importance.head(10).iterrows():
                    # Identify if it's a symptom feature
                    feature_type = ""
                    for symptom in SYMPTOM_CATEGORIES:
                        if symptom in row['feature']:
                            feature_type = f" [{symptom}]"
                            break
                    print(f"      {row['feature']}{feature_type}: {row['importance']:.4f}")
                
                results['feature_importance'] = feature_importance
    
    else:  # Regression
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_score = -np.inf
        best_model = None
        best_model_name = None
        
        for name, model in models.items():
            try:
                # Cross-validation with R2 score
                scores = cross_val_score(model, X_clean, y, cv=tscv, scoring='r2')
                mean_score = scores.mean()
                
                print(f"   {name}: R²={mean_score:.4f} (+/- {scores.std():.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"   {name}: Failed - {str(e)}")
        
        if best_model:
            # Train best model on full data
            best_model.fit(X_clean, y)
            
            # Get predictions
            y_pred = best_model.predict(X_clean)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = np.mean(np.abs(y - y_pred))
            
            # Calculate percentage within ±1 visit
            within_one = np.mean(np.abs(y - y_pred) <= 1) * 100
            
            results = {
                'model': best_model_name,
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'within_one_visit': within_one,
                'best_model': best_model
            }
            
            print(f"\n   ✅ Best Model: {best_model_name}")
            print(f"      R² Score: {r2:.4f}")
            print(f"      RMSE: {rmse:.4f}")
            print(f"      MAE: {mae:.4f}")
            print(f"      Within ±1 visit: {within_one:.1f}%")
    
    return results

# Main training loop
def main():
    """Train models on all available datasets"""
    
    # Define datasets to train
    datasets = {
        # General visit prediction
        'binary_visits': {
            'file': 'dataset_binary_visits.csv',
            'type': 'classification',
            'target': 'target',
            'description': 'Binary Visit Prediction (Visit/No Visit)'
        },
        'multiclass_visits': {
            'file': 'dataset_multiclass_visits.csv',
            'type': 'classification',
            'target': 'target',
            'description': 'Multi-class Visit Categories'
        },
        'risk_based': {
            'file': 'dataset_risk_based.csv',
            'type': 'classification',
            'target': 'target',
            'description': 'Risk-based Classification'
        },
        'regression_visits': {
            'file': 'dataset_regression_visits.csv',
            'type': 'regression',
            'target': 'target',
            'description': 'Visit Count Regression'
        },
        
        # Symptom-specific predictions
        'dominant_symptom': {
            'file': 'dataset_dominant_symptom.csv',
            'type': 'classification',
            'target': 'target',
            'description': 'Dominant Symptom Classification'
        },
        'symptom_severity': {
            'file': 'dataset_symptom_severity.csv',
            'type': 'classification',
            'target': 'target',
            'description': 'Symptom Severity Classification'
        },
        'respiratory_outbreak': {
            'file': 'dataset_respiratory_outbreak.csv',
            'type': 'classification',
            'target': 'target',
            'description': 'Respiratory Outbreak Detection'
        }
    }
    
    # Add binary symptom predictions for each category
    for symptom in SYMPTOM_CATEGORIES:
        datasets[f'binary_{symptom}'] = {
            'file': f'dataset_binary_{symptom}_present.csv',
            'type': 'classification',
            'target': 'target',
            'description': f'{symptom.title()} Symptom Detection'
        }
    
    all_results = {}
    
    print("\n" + "=" * 80)
    print("TRAINING MODELS ON ALL DATASETS")
    print("=" * 80)
    
    for dataset_name, config in datasets.items():
        try:
            # Load dataset
            df = pd.read_csv(f'medisense/backend/data/final/{config["file"]}')
            
            # Separate features and target
            if 'date' in df.columns:
                df = df.drop(columns=['date'])
            
            X = df.drop(columns=[config['target']])
            y = df[config['target']]
            
            print(f"\n{'='*80}")
            print(f"Dataset: {config['description']}")
            print(f"Shape: {X.shape}")
            print(f"Target type: {config['type']}")
            
            # Train model
            results = train_model_with_smote(
                X, y, 
                model_type=config['type'],
                task_name=config['description']
            )
            
            all_results[dataset_name] = results
            
        except FileNotFoundError:
            print(f"\n⚠️  Dataset {config['file']} not found, skipping...")
        except Exception as e:
            print(f"\n❌ Error training {dataset_name}: {str(e)}")
    
    # Summary report
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    print("\n📊 CLASSIFICATION TASKS:")
    for name, results in all_results.items():
        if 'accuracy' in results:
            print(f"\n{name}:")
            print(f"   Model: {results['model']}")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   F1-Score: {results['f1_score']:.4f}")
            if results.get('auc_roc'):
                print(f"   AUC-ROC: {results['auc_roc']:.4f}")
    
    print("\n📈 REGRESSION TASKS:")
    for name, results in all_results.items():
        if 'r2_score' in results:
            print(f"\n{name}:")
            print(f"   Model: {results['model']}")
            print(f"   R² Score: {results['r2_score']:.4f}")
            print(f"   RMSE: {results['rmse']:.4f}")
            print(f"   Within ±1 visit: {results['within_one_visit']:.1f}%")
    
    # Identify symptom-specific insights
    print("\n" + "=" * 80)
    print("SYMPTOM-SPECIFIC INSIGHTS")
    print("=" * 80)
    
    symptom_performance = {}
    for symptom in SYMPTOM_CATEGORIES:
        key = f'binary_{symptom}'
        if key in all_results and 'f1_score' in all_results[key]:
            symptom_performance[symptom] = all_results[key]['f1_score']
    
    if symptom_performance:
        sorted_symptoms = sorted(symptom_performance.items(), key=lambda x: x[1], reverse=True)
        print("\n🏥 Symptom Detection Performance (F1-Score):")
        for symptom, score in sorted_symptoms:
            print(f"   {symptom:25s}: {score:.4f}")
        
        print(f"\n   Best detected symptom: {sorted_symptoms[0][0]} ({sorted_symptoms[0][1]:.4f})")
        print(f"   Hardest to detect: {sorted_symptoms[-1][0]} ({sorted_symptoms[-1][1]:.4f})")
    
    return all_results

if __name__ == "__main__":
    results = main()
    print("\n✅ Training complete!")
