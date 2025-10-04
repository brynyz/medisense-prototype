"""
Simplified Visit Count Prediction
==================================
This script uses the revised dataset with:
- Only dominant symptom categories (respiratory, digestive)
- Simplified feature set (12 features)
- Two-stage approach for zero-inflated data

Author: MediSense Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score, 
    mean_absolute_error, classification_report
)
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SIMPLIFIED VISIT COUNT PREDICTION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Features: dow, lag features, environmental factors")
print("Target: visit_count")
print("=" * 80)

def load_data():
    """Load the simplified training dataset"""
    print("\n📁 Loading simplified dataset...")
    
    # Try to load the training-ready dataset
    try:
        df = pd.read_csv('medisense/backend/data/final/training_ready_simple.csv')
        print(f"✅ Loaded training_ready_simple.csv: {df.shape}")
    except FileNotFoundError:
        print("⚠️  training_ready_simple.csv not found, loading daily_revised_simple.csv")
        df = pd.read_csv('medisense/backend/data/final/visit/daily_revised_simple.csv')
        # Remove first 7 days for lag features
        df = df.iloc[7:].dropna()
        print(f"✅ Loaded and cleaned: {df.shape}")
    
    return df

def prepare_features(df):
    """Prepare features and target"""
    print("\n🔧 Preparing features...")
    
    # Define feature columns (as specified by user)
    feature_cols = [
        'dow',           # Day of week
        'visits_lag1',   # Yesterday's visits
        'visits_lag7',   # Last week's visits
        'visits_rollmean7',  # 7-day rolling mean
        'resp_lag1',     # Yesterday's respiratory count
        'resp_rollmean7',    # 7-day respiratory rolling mean
        'digest_lag1',   # Yesterday's digestive count
        'digest_rollmean7',  # 7-day digestive rolling mean
        'temp',          # Temperature
        'humidity',      # Humidity
        'rainfall',      # Rainfall
        'pm2_5'          # PM2.5 air quality
    ]
    
    # Check which features are available
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
    
    print(f"✅ Using {len(available_features)} features: {available_features}")
    
    X = df[available_features]
    y = df['visit_count']
    
    # Show target distribution
    print(f"\n📊 Target Distribution:")
    print(f"   Zero-visit days: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"   Days with visits: {(y > 0).sum()} ({(y > 0).mean()*100:.1f}%)")
    print(f"   Max visits: {y.max()}")
    print(f"   Mean visits: {y.mean():.2f}")
    print(f"   Median visits: {y.median():.0f}")
    
    return X, y

def train_two_stage_model(X, y):
    """
    Train a two-stage model:
    Stage 1: Binary classification (visit/no-visit)
    Stage 2: Regression for visit count (only for predicted visit days)
    """
    print("\n" + "="*60)
    print("TWO-STAGE MODEL TRAINING")
    print("="*60)
    
    # Create binary target for Stage 1
    y_binary = (y > 0).astype(int)
    
    # ========================================
    # STAGE 1: Binary Classification
    # ========================================
    print("\n📌 Stage 1: Binary Classification (Visit/No-Visit)")
    print("-" * 50)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    
    # Cross-validation
    cv_scores = cross_val_score(rf_classifier, X, y_binary, cv=tscv, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Train on full data
    rf_classifier.fit(X, y_binary)
    binary_pred = rf_classifier.predict(X)
    binary_acc = accuracy_score(y_binary, binary_pred)
    print(f"   Training Accuracy: {binary_acc:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n   Top 5 Features for Visit Prediction:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"      {row['feature']:20s}: {row['importance']:.3f}")
    
    # ========================================
    # STAGE 2: Regression (for visit days only)
    # ========================================
    print("\n📌 Stage 2: Regression (Visit Count for Predicted Visit Days)")
    print("-" * 50)
    
    # Filter to only days with visits for training regression
    visit_mask = y > 0
    X_visits = X[visit_mask]
    y_visits = y[visit_mask]
    
    print(f"   Training on {len(X_visits)} days with visits")
    print(f"   Visit range: {y_visits.min()}-{y_visits.max()}, Mean: {y_visits.mean():.2f}")
    
    # Try different regression models
    regressors = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'Linear': LinearRegression(),
        'Poisson': PoissonRegressor(max_iter=1000)
    }
    
    best_regressor = None
    best_rmse = float('inf')
    best_name = None
    
    for name, regressor in regressors.items():
        # Cross-validation on visit days only
        cv_scores = cross_val_score(
            regressor, X_visits, y_visits, 
            cv=min(3, len(X_visits)//10),  # Adjust CV folds based on data size
            scoring='neg_mean_squared_error'
        )
        rmse = np.sqrt(-cv_scores.mean())
        print(f"   {name:12s} RMSE: {rmse:.3f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_regressor = regressor
            best_name = name
    
    print(f"\n   Best Regressor: {best_name} (RMSE: {best_rmse:.3f})")
    
    # Train best regressor on all visit days
    best_regressor.fit(X_visits, y_visits)
    
    return rf_classifier, best_regressor, best_name

def evaluate_combined_model(classifier, regressor, X, y):
    """Evaluate the combined two-stage model"""
    print("\n" + "="*60)
    print("COMBINED MODEL EVALUATION")
    print("="*60)
    
    # Stage 1: Predict which days have visits
    visit_predictions = classifier.predict(X)
    
    # Stage 2: Predict visit count for predicted visit days
    final_predictions = np.zeros(len(X))
    visit_indices = np.where(visit_predictions == 1)[0]
    
    if len(visit_indices) > 0:
        X_predicted_visits = X.iloc[visit_indices]
        count_predictions = regressor.predict(X_predicted_visits)
        final_predictions[visit_indices] = count_predictions
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, final_predictions))
    mae = mean_absolute_error(y, final_predictions)
    r2 = r2_score(y, final_predictions)
    
    # Accuracy within ±1 visit
    within_1 = np.mean(np.abs(y - final_predictions) <= 1)
    within_2 = np.mean(np.abs(y - final_predictions) <= 2)
    
    # Separate metrics for zero and non-zero days
    zero_mask = y == 0
    nonzero_mask = y > 0
    
    zero_accuracy = np.mean(final_predictions[zero_mask] == 0) if zero_mask.sum() > 0 else 0
    nonzero_recall = np.mean(final_predictions[nonzero_mask] > 0) if nonzero_mask.sum() > 0 else 0
    
    print(f"\n📊 Overall Performance:")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   MAE: {mae:.3f}")
    print(f"   R² Score: {r2:.3f}")
    print(f"   Within ±1 visit: {within_1:.1%}")
    print(f"   Within ±2 visits: {within_2:.1%}")
    
    print(f"\n📊 Detailed Performance:")
    print(f"   Zero-day accuracy: {zero_accuracy:.1%}")
    print(f"   Visit-day recall: {nonzero_recall:.1%}")
    
    # Show some example predictions
    print(f"\n📋 Sample Predictions (first 10 non-zero days):")
    nonzero_indices = np.where(y > 0)[0][:10]
    for idx in nonzero_indices:
        actual = y.iloc[idx]
        predicted = final_predictions[idx]
        error = predicted - actual
        print(f"   Day {idx}: Actual={actual:.0f}, Predicted={predicted:.1f}, Error={error:+.1f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'within_1': within_1,
        'within_2': within_2,
        'zero_accuracy': zero_accuracy,
        'nonzero_recall': nonzero_recall
    }

def main():
    """Main execution"""
    import os
    os.makedirs('medisense/backend/models/simplified', exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train two-stage model
    classifier, regressor, regressor_name = train_two_stage_model(X, y)
    
    # Evaluate combined model
    metrics = evaluate_combined_model(classifier, regressor, X, y)
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # joblib.dump(classifier, 'medisense/backend/models/simplified/binary_classifier.pkl')
    # print("✅ Saved: binary_classifier.pkl")
    
    # joblib.dump(regressor, f'medisense/backend/models/simplified/regressor_{regressor_name.lower()}.pkl')
    # print(f"✅ Saved: regressor_{regressor_name.lower()}.pkl")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'features': list(X.columns),
        'n_features': len(X.columns),
        'n_samples': len(X),
        'zero_inflation': (y == 0).mean(),
        'regressor_type': regressor_name,
        'metrics': metrics
    }
    
    # with open('medisense/backend/models/simplified/results.json', 'w') as f:
    #     json.dump(results, f, indent=2, default=float)
    # print("✅ Saved: results.json")
    
    # Create summary report
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\n✨ Key Results:")
    print(f"   • Accuracy within ±1 visit: {metrics['within_1']:.1%}")
    print(f"   • Zero-day accuracy: {metrics['zero_accuracy']:.1%}")
    print(f"   • Visit-day recall: {metrics['nonzero_recall']:.1%}")
    print(f"   • RMSE: {metrics['rmse']:.3f}")
    print(f"   • R² Score: {metrics['r2']:.3f}")
    
    print(f"\n📁 Output Files:")
    print(f"   • models/simplified/binary_classifier.pkl")
    print(f"   • models/simplified/regressor_{regressor_name.lower()}.pkl")
    print(f"   • models/simplified/results.json")
    
    return results

if __name__ == "__main__":
    main()
