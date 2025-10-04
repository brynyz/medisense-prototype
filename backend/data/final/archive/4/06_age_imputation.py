import pandas as pd
import numpy as np

print("🔧 FIXING AGE HANDLING IN DAILY AGGREGATES")
print("=" * 50)

# Load the current daily complete dataset
df = pd.read_csv('medisense/backend/data/final/daily_complete_fixed.csv')

print(f"📊 Current Age Analysis:")
print(f"   Total rows: {len(df)}")
print(f"   Age = 20 count: {(df['age'] == 20).sum()}")
print(f"   Zero visit days: {(df['visit_count'] == 0).sum()}")
print(f"   Age distribution:\n{df['age'].value_counts().head(10)}")

# SOLUTION 1: Remove age from zero-visit days (RECOMMENDED)
def fix_age_solution_1(df):
    """Remove age feature for zero-visit days"""
    df_fixed = df.copy()
    
    # Set age to NaN for zero-visit days (will be handled by feature engineering)
    zero_visit_mask = df_fixed['visit_count'] == 0
    df_fixed.loc[zero_visit_mask, 'age'] = np.nan
    
    print(f"\n✅ SOLUTION 1 - Remove age for zero-visit days:")
    print(f"   Age set to NaN for {zero_visit_mask.sum()} zero-visit days")
    print(f"   Remaining age = 20 count: {(df_fixed['age'] == 20).sum()}")
    
    return df_fixed

# SOLUTION 2: Use rolling average age (ALTERNATIVE)
def fix_age_solution_2(df):
    """Use rolling average age from recent visit days"""
    df_fixed = df.copy()
    
    # Calculate rolling average age from visit days only
    visit_days = df_fixed[df_fixed['visit_count'] > 0]['age']
    rolling_avg_age = visit_days.rolling(window=30, min_periods=1).mean()
    
    # Create a mapping of dates to rolling average ages
    visit_dates = df_fixed[df_fixed['visit_count'] > 0]['date']
    age_mapping = dict(zip(visit_dates, rolling_avg_age))
    
    # Fill zero-visit days with rolling average
    zero_visit_mask = df_fixed['visit_count'] == 0
    for idx in df_fixed[zero_visit_mask].index:
        date = df_fixed.loc[idx, 'date']
        # Find the most recent visit day's rolling average
        recent_avg = 20  # fallback
        for visit_date, avg_age in age_mapping.items():
            if visit_date <= date:
                recent_avg = avg_age
        df_fixed.loc[idx, 'age'] = recent_avg
    
    print(f"\n✅ SOLUTION 2 - Use rolling average age:")
    print(f"   Updated {zero_visit_mask.sum()} zero-visit days with rolling averages")
    print(f"   New age distribution:\n{df_fixed['age'].value_counts().head(5)}")
    
    return df_fixed

# SOLUTION 3: Create separate age features (ADVANCED)
def fix_age_solution_3(df):
    """Create separate age features for visit vs non-visit days"""
    df_fixed = df.copy()
    
    # Create separate age features
    df_fixed['age_on_visit_days'] = df_fixed['age'].where(df_fixed['visit_count'] > 0, np.nan)
    df_fixed['has_age_data'] = df_fixed['visit_count'] > 0
    
    # Remove the original problematic age column
    df_fixed = df_fixed.drop('age', axis=1)
    
    print(f"\n✅ SOLUTION 3 - Separate age features:")
    print(f"   Created 'age_on_visit_days' and 'has_age_data' features")
    print(f"   Non-null age_on_visit_days: {df_fixed['age_on_visit_days'].notna().sum()}")
    
    return df_fixed

# Test all solutions
print(f"\n🧪 TESTING SOLUTIONS:")

# Solution 1 (RECOMMENDED)
df_solution1 = fix_age_solution_1(df)

# Solution 2 
df_solution2 = fix_age_solution_2(df)

# Solution 3
df_solution3 = fix_age_solution_3(df)

# Save the recommended solution
df_solution1.to_csv('medisense/backend/data/final/daily_complete_age_fixed.csv', index=False)

print(f"\n💡 RECOMMENDATION:")
print(f"   Use SOLUTION 1 - Remove age for zero-visit days")
print(f"   ✅ Prevents feature leakage")
print(f"   ✅ Maintains data integrity") 
print(f"   ✅ Let your comprehensive optimization handle missing values properly")
print(f"")
print(f"   Saved as: daily_complete_age_fixed.csv")
print(f"   Update your comprehensive_dataset_optimization.py to use this file!")

print(f"\n🎯 WHY SOLUTION 1 IS BEST:")
print(f"   • No artificial age data on zero-visit days")
print(f"   • Your comprehensive optimization will handle NaN values correctly")
print(f"   • Prevents the model from learning 'age=20 means no visits'")
print(f"   • Maintains the true age distribution from actual visits")
