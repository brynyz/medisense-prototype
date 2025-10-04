"""
Diagnose Regression Performance Issues
======================================
Check why R² scores are negative
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

print("=" * 80)
print("DIAGNOSING REGRESSION PERFORMANCE")
print("=" * 80)

# Load regression dataset
df = pd.read_csv('medisense/backend/data/final/dataset_regression_visits.csv')
print(f"\nDataset shape: {df.shape}")

# Check target distribution
y = df['target']
print(f"\n📊 Target (Visit Count) Statistics:")
print(f"   Mean: {y.mean():.2f}")
print(f"   Std: {y.std():.2f}")
print(f"   Min: {y.min()}")
print(f"   Max: {y.max()}")
print(f"   Zeros: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")

# Value counts
print(f"\n📈 Visit Count Distribution:")
value_counts = y.value_counts().sort_index()
for value, count in value_counts.head(10).items():
    print(f"   {value} visits: {count} days ({count/len(y)*100:.1f}%)")

# Check for temporal patterns
df['date'] = pd.to_datetime(df['date'], errors='coerce')
if df['date'].notna().any():
    df = df.sort_values('date')
    
    # Plot visit counts over time
    print(f"\n📅 Temporal Analysis:")
    
    # Split into train/test (80/20)
    split_idx = int(len(df) * 0.8)
    train_y = y[:split_idx]
    test_y = y[split_idx:]
    
    print(f"   Training period: {df['date'].iloc[0]} to {df['date'].iloc[split_idx-1]}")
    print(f"   Testing period: {df['date'].iloc[split_idx]} to {df['date'].iloc[-1]}")
    
    print(f"\n   Training set:")
    print(f"     Mean visits: {train_y.mean():.2f}")
    print(f"     Std: {train_y.std():.2f}")
    print(f"     Zero days: {(train_y == 0).mean()*100:.1f}%")
    
    print(f"\n   Testing set:")
    print(f"     Mean visits: {test_y.mean():.2f}")
    print(f"     Std: {test_y.std():.2f}")
    print(f"     Zero days: {(test_y == 0).mean()*100:.1f}%")
    
    # Check if there's a distribution shift
    if abs(train_y.mean() - test_y.mean()) > 1:
        print(f"\n⚠️  WARNING: Significant distribution shift detected!")
        print(f"   Train mean: {train_y.mean():.2f} vs Test mean: {test_y.mean():.2f}")
    
    # Simple baseline predictions
    print(f"\n🎯 Baseline Predictions:")
    
    # Baseline 1: Always predict training mean
    baseline_mean = np.full(len(test_y), train_y.mean())
    r2_mean = r2_score(test_y, baseline_mean)
    rmse_mean = np.sqrt(mean_squared_error(test_y, baseline_mean))
    
    print(f"   Predict mean ({train_y.mean():.2f}):")
    print(f"     R²: {r2_mean:.4f}")
    print(f"     RMSE: {rmse_mean:.4f}")
    
    # Baseline 2: Always predict zero (most common)
    baseline_zero = np.zeros(len(test_y))
    r2_zero = r2_score(test_y, baseline_zero)
    rmse_zero = np.sqrt(mean_squared_error(test_y, baseline_zero))
    
    print(f"   Predict zero:")
    print(f"     R²: {r2_zero:.4f}")
    print(f"     RMSE: {rmse_zero:.4f}")
    
    # Baseline 3: Last value carry forward
    baseline_last = np.full(len(test_y), train_y.iloc[-1])
    r2_last = r2_score(test_y, baseline_last)
    rmse_last = np.sqrt(mean_squared_error(test_y, baseline_last))
    
    print(f"   Last value ({train_y.iloc[-1]}):")
    print(f"     R²: {r2_last:.4f}")
    print(f"     RMSE: {rmse_last:.4f}")

# Check feature quality
X = df.drop(columns=['target', 'date'], errors='ignore')

# Remove current symptom counts
symptom_current = ['respiratory_count', 'digestive_count', 'pain_musculoskeletal_count',
                  'dermatological_trauma_count', 'neuro_psych_count', 
                  'systemic_infectious_count', 'cardiovascular_chronic_count', 'other_count']
X = X.drop(columns=[col for col in symptom_current if col in X.columns])

print(f"\n📊 Feature Analysis:")
print(f"   Total features: {X.shape[1]}")
print(f"   Missing values: {X.isnull().sum().sum()}")

# Check correlation with target
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print(f"\n   Top 10 correlated features:")
for feat, corr in correlations.head(10).items():
    print(f"     {feat}: {corr:.4f}")

print(f"\n💡 Insights:")
if (y == 0).mean() > 0.8:
    print("   - Extreme class imbalance (>80% zeros)")
    print("   - Consider zero-inflated models or two-stage approach")
    print("   - R² can be negative when model performs worse than mean baseline")

if abs(train_y.mean() - test_y.mean()) > 1:
    print("   - Distribution shift between train and test")
    print("   - Model may not generalize well to test period")
    print("   - Consider time-aware features or different split strategy")

print("\n✅ Diagnosis complete!")
