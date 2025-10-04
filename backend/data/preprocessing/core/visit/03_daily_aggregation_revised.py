import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("REVISED DAILY AGGREGATION - DOMINANT SYMPTOMS ONLY")
print("=" * 70)
print("Focus: Respiratory and Digestive as dominant categories")
print("Simplified feature set for visit count prediction")
print("=" * 70)

# Load your datasets
medical_df = pd.read_csv('medisense/backend/data/cleaned/cleaned_colummns.csv')
weather_df = pd.read_csv('medisense/backend/data/raw/historical_weather.csv')  
aqi_df = pd.read_csv('medisense/backend/data/raw/daily_aqi.csv')

print("\n📊 Data Loading:")
print(f"Medical records: {len(medical_df)} rows")
print(f"Weather data: {len(weather_df)} rows")
print(f"AQI data: {len(aqi_df)} rows")

# Fix date formats - NORMALIZE TO DATE ONLY (NO TIME)
medical_df['date_cleaned'] = pd.to_datetime(medical_df['date_cleaned'], format='%m-%d-%Y', errors='coerce')
weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
weather_df['date'] = pd.to_datetime(weather_df['date'])
medical_df['date_cleaned'] = medical_df['date_cleaned'].dt.date
medical_df['date_cleaned'] = pd.to_datetime(medical_df['date_cleaned'])

if 'date' in aqi_df.columns:
    aqi_df['date'] = pd.to_datetime(aqi_df['date']).dt.date
    aqi_df['date'] = pd.to_datetime(aqi_df['date'])

print(f"\n📅 Date ranges:")
print(f"Medical: {medical_df['date_cleaned'].min()} to {medical_df['date_cleaned'].max()}")
print(f"Weather: {weather_df['date'].min()} to {weather_df['date'].max()}")

def categorize_dominant_symptoms(symptoms_str):
    """
    REVISED: Only categorize into dominant categories (respiratory and digestive)
    This prevents dataset inflation and focuses on the most common symptom types
    """
    if pd.isna(symptoms_str) or symptoms_str == '':
        return {'respiratory': 0, 'digestive': 0}
    
    symptoms_lower = symptoms_str.lower()
    
    # Respiratory symptoms (most common category)
    respiratory_symptoms = [
        'cold', 'cough', 'asthma', 'runny nose', 'sore throat', 'itchy throat', 
        'auri', 'shortness of breath', 'hyperventilation', 'earache', 'nosebleed',
        'fever'  # Often associated with respiratory infections
    ]
    
    # Digestive symptoms (second most common category)
    digestive_symptoms = [
        'stomach ache', 'hyperacidity', 'lbm', 'diarrhea', 'vomiting', 
        'epigastric pain', 'dry mouth', 'nausea', 'abdominal pain'
    ]
    
    # Count matches for each category
    counts = {
        'respiratory': sum(1 for sym in respiratory_symptoms if sym in symptoms_lower),
        'digestive': sum(1 for sym in digestive_symptoms if sym in symptoms_lower)
    }
    
    return counts

# ==========================================
# CREATE DAILY AGGREGATES WITH COMPLETE DATE RANGE
# ==========================================

print("\n📊 Creating Daily Aggregates...")

# Create complete date range
start_date = medical_df['date_cleaned'].min()
end_date = medical_df['date_cleaned'].max()
complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Create base dataframe with all dates
daily_df = pd.DataFrame({'date': complete_dates})

# Add day of week (0=Monday, 6=Sunday)
daily_df['day_of_week'] = daily_df['date'].dt.dayofweek

# Aggregate medical data by day
daily_medical = medical_df.groupby('date_cleaned').agg({
    'normalized_symptoms': lambda x: ','.join(x.dropna())
}).reset_index()

# Add visit count
visit_counts = medical_df.groupby('date_cleaned').size().reset_index(name='visit_count')
daily_medical = daily_medical.merge(visit_counts, on='date_cleaned')

# Add dominant symptom categories
symptom_categories = daily_medical['normalized_symptoms'].apply(categorize_dominant_symptoms)
daily_medical['respiratory_count'] = [cat['respiratory'] for cat in symptom_categories]
daily_medical['digestive_count'] = [cat['digestive'] for cat in symptom_categories]

# Merge with complete date range
daily_df = daily_df.merge(daily_medical, left_on='date', right_on='date_cleaned', how='left')

# Fill missing values with zeros for counts
daily_df['visit_count'] = daily_df['visit_count'].fillna(0)
daily_df['respiratory_count'] = daily_df['respiratory_count'].fillna(0)
daily_df['digestive_count'] = daily_df['digestive_count'].fillna(0)

print(f"✅ Daily aggregation complete: {len(daily_df)} days")
print(f"   Days with visits: {(daily_df['visit_count'] > 0).sum()}")
print(f"   Zero-visit days: {(daily_df['visit_count'] == 0).sum()}")

# ==========================================
# ADD LAG FEATURES (CRITICAL FOR PREDICTION)
# ==========================================

print("\n📈 Adding Lag Features...")

# Sort by date for proper lag calculation
daily_df = daily_df.sort_values('date').reset_index(drop=True)

# Visit count lags
daily_df['visits_lag1'] = daily_df['visit_count'].shift(1)
daily_df['visits_lag7'] = daily_df['visit_count'].shift(7)
daily_df['visits_rollmean7'] = daily_df['visit_count'].rolling(window=7, min_periods=1).mean()

# Respiratory symptom lags
daily_df['resp_lag1'] = daily_df['respiratory_count'].shift(1)
daily_df['resp_rollmean7'] = daily_df['respiratory_count'].rolling(window=7, min_periods=1).mean()

# Digestive symptom lags
daily_df['digest_lag1'] = daily_df['digestive_count'].shift(1)
daily_df['digest_rollmean7'] = daily_df['digestive_count'].rolling(window=7, min_periods=1).mean()

print("✅ Lag features added")

# ==========================================
# MERGE ENVIRONMENTAL DATA
# ==========================================

print("\n🌤️ Merging Environmental Data...")

# Select only the environmental features we need
weather_features = ['temperature_2m_mean', 'relative_humidity_2m_mean', 'precipitation_sum']
weather_subset = weather_df[['date'] + weather_features].copy()

# Rename columns to match user requirements
weather_subset = weather_subset.rename(columns={
    'temperature_2m_mean': 'temp',
    'relative_humidity_2m_mean': 'humidity',
    'precipitation_sum': 'rainfall'
})

# Merge weather data
daily_df = daily_df.merge(weather_subset, on='date', how='left')

# Merge AQI data (PM2.5 only)
if 'pm2_5_mean' in aqi_df.columns:
    aqi_subset = aqi_df[['date', 'pm2_5_mean']].copy()
    aqi_subset = aqi_subset.rename(columns={'pm2_5_mean': 'pm2_5'})
    daily_df = daily_df.merge(aqi_subset, on='date', how='left')
else:
    daily_df['pm2_5'] = np.nan

print("✅ Environmental data merged")

# Fill missing environmental values with median
env_cols = ['temp', 'humidity', 'rainfall', 'pm2_5']
for col in env_cols:
    if col in daily_df.columns:
        median_val = daily_df[col].median()
        daily_df[col] = daily_df[col].fillna(median_val)
        print(f"   {col}: filled missing with median {median_val:.2f}")

# ==========================================
# FINAL FEATURE SELECTION
# ==========================================

print("\n🎯 Final Feature Selection (Simplified):")

# Select only the features specified by user (excluding date and target)
final_features = [
    'day_of_week',  # dow
    'visits_lag1',
    'visits_lag7', 
    'visits_rollmean7',
    'resp_lag1',
    'resp_rollmean7',
    'digest_lag1',
    'digest_rollmean7',
    'temp',
    'humidity',
    'rainfall',
    'pm2_5',
    'visit_count'  # Keep target for dataset creation, will separate later
]

# Ensure all features exist
available_features = [f for f in final_features if f in daily_df.columns]
missing_features = [f for f in final_features if f not in daily_df.columns]

if missing_features:
    print(f"⚠️  Missing features: {missing_features}")

daily_final = daily_df[available_features].copy()

# Rename day_of_week to dow for consistency
daily_final = daily_final.rename(columns={'day_of_week': 'dow'})

print(f"✅ Final dataset shape: {daily_final.shape}")
print(f"   Features: {list(daily_final.columns)}")

# ==========================================
# DATA QUALITY CHECK
# ==========================================

print("\n🔍 Data Quality Check:")

# Check for NaN values
nan_counts = daily_final.isna().sum()
if nan_counts.sum() > 0:
    print("NaN values per column:")
    for col, count in nan_counts[nan_counts > 0].items():
        print(f"   {col}: {count} ({count/len(daily_final)*100:.1f}%)")
else:
    print("✅ No NaN values in dataset")

# Show feature statistics
print("\n📊 Feature Statistics:")
numeric_cols = daily_final.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'dow':
        mean_val = daily_final[col].mean()
        std_val = daily_final[col].std()
        min_val = daily_final[col].min()
        max_val = daily_final[col].max()
        print(f"   {col:20s}: mean={mean_val:7.2f}, std={std_val:7.2f}, range=[{min_val:.1f}, {max_val:.1f}]")

# ==========================================
# SAVE DATASETS
# ==========================================

print("\n💾 Saving Datasets...")

# Save the complete dataset
daily_final.to_csv('medisense/backend/data/final/daily_revised_simple.csv', index=False)
print(f"✅ Saved: daily_revised_simple.csv")

# Create training dataset (remove first 7 days due to lags)
training_df = daily_final.iloc[7:].copy()  # Skip first week for lag features
training_df = training_df.dropna()  # Remove any remaining NaN rows

# Split features and target
feature_cols = [col for col in training_df.columns if col != 'visit_count']
X = training_df[feature_cols]
y = training_df['visit_count']

print(f"✅ Features ({len(feature_cols)}): {feature_cols}")
print(f"✅ Target: visit_count")

# Save training-ready dataset (features + target)
training_df.to_csv('medisense/backend/data/final/training_ready_simple.csv', index=False)
print(f"✅ Saved: training_ready_simple.csv")

# Save features-only dataset for easy loading
X.to_csv('medisense/backend/data/final/features_only_simple.csv', index=False)
print(f"✅ Saved: features_only_simple.csv")

# ==========================================
# SUMMARY REPORT
# ==========================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"\n📊 Final Dataset Summary:")
print(f"   Total days: {len(daily_final)}")
print(f"   Training days: {len(training_df)}")
print(f"   Features: {len(final_features)}")
print(f"   Target: visit_count")

print(f"\n🎯 Target Distribution:")
print(f"   Zero-visit days: {(training_df['visit_count'] == 0).sum()} ({(training_df['visit_count'] == 0).mean()*100:.1f}%)")
print(f"   Days with visits: {(training_df['visit_count'] > 0).sum()} ({(training_df['visit_count'] > 0).mean()*100:.1f}%)")
print(f"   Max visits in a day: {training_df['visit_count'].max()}")
print(f"   Mean visits per day: {training_df['visit_count'].mean():.2f}")

print(f"\n✨ Key Improvements:")
print(f"   1. Focused on dominant symptom categories only (respiratory, digestive)")
print(f"   2. Simplified feature set to 12 predictors")
print(f"   3. Added proper lag features for temporal patterns")
print(f"   4. Included essential environmental factors")
print(f"   5. No data leakage - using only lagged symptom counts")

print(f"\n📁 Output Files:")
print(f"   1. daily_revised_simple.csv - Complete daily dataset")
print(f"   2. training_ready_simple.csv - Ready for model training")

print("\n" + "=" * 70)
