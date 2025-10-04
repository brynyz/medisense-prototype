import pandas as pd
import numpy as np

# Quick fix for the missing value handling issue
df = pd.read_csv('medisense/backend/data/final/initial dataset/daily_complete.csv')

print("🔧 FIXING MISSING VALUES")
print("=" * 50)

# Handle environmental columns properly
env_cols = [col for col in df.columns if any(x in col.lower() for x in ['temp', 'humid', 'pm', 'aqi', 'wind', 'precip'])]

for col in env_cols:
    if col in df.columns:
        before_missing = df[col].isna().sum()
        
        # Forward fill first
        df[col] = df[col].fillna(method='ffill', limit=3)
        
        # Handle based on data type
        if df[col].dtype == 'object':
            # Categorical - use most frequent value
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'unknown'
            df[col] = df[col].fillna(fill_val)
        else:
            # Numeric - use median
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].fillna(df[col].median())
        
        after_missing = df[col].isna().sum()
        if before_missing > 0:
            print(f"   {col}: {before_missing} → {after_missing} missing values")

# Save the fixed dataset
df.to_csv('medisense/backend/data/final/daily_complete_fixed.csv', index=False)
print(f"\n✅ Fixed dataset saved as daily_complete_fixed.csv")
print(f"   Total remaining missing values: {df.isnull().sum().sum()}")

print("\n💡 Now update your comprehensive_dataset_optimization.py:")
print("   Change line 17 to: df = pd.read_csv('medisense/backend/data/final/daily_complete_fixed.csv')")
