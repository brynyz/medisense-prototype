"""
Fixed Final Model Training - Handles XGBoost Multiclass Properly
================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, r2_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FINAL MODEL TRAINING - FIXED VERSION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

def train_dominant_symptom_classification():
    """Train dominant symptom classification with proper label encoding"""
    
    print("\n" + "="*80)
    print("DOMINANT SYMPTOM CLASSIFICATION (FIXED)")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv('medisense/backend/data/final/optimized/dataset_dominant_symptom_optimized.csv')
    print(f"Dataset shape: {df.shape}")
    
    X = df.drop(columns=['target', 'date'], errors='ignore')
    y = df['target']
    
    print(f"Features: {X.shape[1]}")
    print(f"Unique symptom classes: {y.nunique()}")
    
    # Show class distribution
    class_dist = pd.Series(y).value_counts()
    print("\nClass distribution:")
    for symptom, count in class_dist.items():
        print(f"  {symptom}: {count}")
    
    # Create label encoder for all models
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nLabel encoding mapping:")
    for i, label in enumerate(le.classes_):
        print(f"  {i} -> {label}")
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Train Random Forest (works with string labels)
    print("\n" + "-"*60)
    print("RANDOM FOREST TRAINING")
    print("-"*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    
    rf_scores = {'acc': [], 'f1': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if len(np.unique(y_val)) < 2:
            print(f"Fold {fold}: Skipped (only one class)")
            continue
        
        # Train with original string labels
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='weighted', zero_division=0
        )
        
        rf_scores['acc'].append(acc)
        rf_scores['f1'].append(f1)
        
        print(f"Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}")
    
    rf_avg_acc = np.mean(rf_scores['acc'])
    rf_avg_f1 = np.mean(rf_scores['f1'])
    print(f"Average: Acc={rf_avg_acc:.3f}, F1={rf_avg_f1:.3f}")
    
    # Train XGBoost (needs numeric labels)
    print("\n" + "-"*60)
    print("XGBOOST TRAINING (WITH LABEL ENCODING)")
    print("-"*60)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',  # Use mlogloss for multiclass
        objective='multi:softmax',  # Explicitly set multiclass objective
        num_class=len(le.classes_)  # Specify number of classes
    )
    
    xgb_scores = {'acc': [], 'f1': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_encoded = y_encoded[train_idx]
        y_val_encoded = y_encoded[val_idx]
        y_val_original = y.iloc[val_idx]
        
        if len(np.unique(y_val_encoded)) < 2:
            print(f"Fold {fold}: Skipped (only one class)")
            continue
        
        try:
            # Train with encoded labels
            xgb_model.fit(X_train, y_train_encoded)
            y_pred_encoded = xgb_model.predict(X_val)
            
            # Convert predictions back to original labels
            y_pred = le.inverse_transform(y_pred_encoded.astype(int))
            
            acc = accuracy_score(y_val_original, y_pred)
            _, _, f1, _ = precision_recall_fscore_support(
                y_val_original, y_pred, average='weighted', zero_division=0
            )
            
            xgb_scores['acc'].append(acc)
            xgb_scores['f1'].append(f1)
            
            print(f"Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}")
            
        except Exception as e:
            print(f"Fold {fold}: Error - {str(e)}")
            continue
    
    if xgb_scores['acc']:
        xgb_avg_acc = np.mean(xgb_scores['acc'])
        xgb_avg_f1 = np.mean(xgb_scores['f1'])
        print(f"Average: Acc={xgb_avg_acc:.3f}, F1={xgb_avg_f1:.3f}")
    else:
        xgb_avg_acc = 0
        xgb_avg_f1 = 0
        print("XGBoost training failed on all folds")
    
    # Determine best model
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    print(f"Random Forest: Acc={rf_avg_acc:.3f}, F1={rf_avg_f1:.3f}")
    print(f"XGBoost:       Acc={xgb_avg_acc:.3f}, F1={xgb_avg_f1:.3f}")
    
    if rf_avg_f1 >= xgb_avg_f1:
        print("\n🏆 Best Model: Random Forest")
        best_model = rf_model
        best_model_name = "RandomForest"
        best_f1 = rf_avg_f1
        
        # Train final model on full dataset
        print("Training final Random Forest model on full dataset...")
        best_model.fit(X, y)
        
    else:
        print("\n🏆 Best Model: XGBoost")
        best_model = xgb_model
        best_model_name = "XGBoost"
        best_f1 = xgb_avg_f1
        
        # Train final model on full dataset with encoded labels
        print("Training final XGBoost model on full dataset...")
        best_model.fit(X, y_encoded)
        
        # Save label encoder for XGBoost
        joblib.dump(le, 'medisense/backend/models/dominant_symptom_label_encoder.pkl')
        print("✅ Label encoder saved")
    
    # Save the best model
    model_path = f'medisense/backend/models/dominant_symptom_{best_model_name.lower()}_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Create confusion matrix for best model
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    if best_model_name == "XGBoost":
        y_pred_final = le.inverse_transform(best_model.predict(X).astype(int))
    else:
        y_pred_final = best_model.predict(X)
    
    final_acc = accuracy_score(y, y_pred_final)
    final_report = classification_report(y, y_pred_final, zero_division=0)
    
    print(f"Final Accuracy on Full Dataset: {final_acc:.3f}")
    print("\nClassification Report:")
    print(final_report)
    
    # Save results
    results = {
        'best_model': best_model_name,
        'random_forest': {
            'avg_accuracy': rf_avg_acc,
            'avg_f1': rf_avg_f1,
            'fold_scores': rf_scores
        },
        'xgboost': {
            'avg_accuracy': xgb_avg_acc,
            'avg_f1': xgb_avg_f1,
            'fold_scores': xgb_scores
        },
        'final_accuracy': final_acc,
        'class_distribution': class_dist.to_dict()
    }
    
    with open('medisense/backend/data/evaluation/dominant_symptom_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Results saved to: medisense/backend/data/evaluation/dominant_symptom_results.json")
    
    return results

def main():
    """Run the fixed training for dominant symptom classification"""
    
    import os
    os.makedirs('medisense/backend/models', exist_ok=True)
    os.makedirs('medisense/backend/data/evaluation', exist_ok=True)
    
    results = train_dominant_symptom_classification()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    print("\n📊 FINAL SUMMARY:")
    print(f"  Best Model: {results['best_model']}")
    print(f"  Cross-Validation Accuracy: {results[results['best_model'].lower()]['avg_accuracy']:.1%}")
    print(f"  Cross-Validation F1-Score: {results[results['best_model'].lower()]['avg_f1']:.3f}")
    print(f"  Final Accuracy: {results['final_accuracy']:.1%}")
    
    print("\n✅ All models and results saved successfully!")

if __name__ == "__main__":
    main()
