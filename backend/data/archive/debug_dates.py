import pandas as pd
from datetime import datetime

# Load the data
medical_df = pd.read_csv('medisense/backend/data/cleaned/cleaned_colummns.csv')
weather_df = pd.read_csv('medisense/backend/data/environment/historical_weather.csv')

print("🔍 DATE DEBUGGING:")
print("="*50)

# Fix dates
medical_df['date_cleaned'] = pd.to_datetime(medical_df['date_cleaned'], format='%m-%d-%Y', errors='coerce')
weather_df['date'] = pd.to_datetime(weather_df['date'])
if weather_df['date'].dt.tz is not None:
    weather_df['date'] = weather_df['date'].dt.tz_localize(None)

# Check date ranges
print(f"Medical data:")
print(f"  Date range: {medical_df['date_cleaned'].min()} to {medical_df['date_cleaned'].max()}")
print(f"  Total unique dates: {medical_df['date_cleaned'].nunique()}")
print(f"  Sample dates: {medical_df['date_cleaned'].head().tolist()}")

print(f"\nWeather data:")
print(f"  Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")
print(f"  Total unique dates: {weather_df['date'].nunique()}")
print(f"  Sample dates: {weather_df['date'].head().tolist()}")

# Check overlap
medical_dates = set(medical_df['date_cleaned'].dt.date)
weather_dates = set(weather_df['date'].dt.date)
overlap = medical_dates & weather_dates

print(f"\nOverlap Analysis:")
print(f"  Medical unique dates: {len(medical_dates)}")
print(f"  Weather unique dates: {len(weather_dates)}")
print(f"  Overlapping dates: {len(overlap)}")

if len(overlap) > 0:
    print(f"  Sample overlapping dates: {list(overlap)[:5]}")
else:
    print(f"  NO OVERLAP FOUND!")
    print(f"  Medical date examples: {list(medical_dates)[:5]}")
    print(f"  Weather date examples: {list(weather_dates)[:5]}")

# Test a simple merge
test_medical = medical_df[['date_cleaned']].drop_duplicates().head(10)
test_merge = test_medical.merge(weather_df, left_on='date_cleaned', right_on='date', how='left')
print(f"\nTest merge result:")
print(f"  Before merge: {len(test_medical)} medical dates")
print(f"  After merge: {len(test_merge)} rows")
print(f"  Non-null temperature values: {test_merge['temperature_2m_mean'].notna().sum()}")

print("\nTest merge sample:")
print(test_merge[['date_cleaned', 'date', 'temperature_2m_mean']].head())