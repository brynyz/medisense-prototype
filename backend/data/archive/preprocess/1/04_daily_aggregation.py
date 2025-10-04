import pandas as pd
import numpy as np
from datetime import datetime

# Load your datasets
medical_df = pd.read_csv('medisense/backend/data/cleaned/cleaned_colummns.csv')
weather_df = pd.read_csv('medisense/backend/data/environment/historical_weather.csv')  
aqi_df = pd.read_csv('medisense/backend/data/environment/daily_aqi.csv')

print("🔍 DEBUGGING LOADED DATA:")
print(f"Weather columns: {list(weather_df.columns)}")
print(f"Weather shape: {weather_df.shape}")

# Fix date formats - NORMALIZE TO DATE ONLY (NO TIME)
medical_df['date_cleaned'] = pd.to_datetime(medical_df['date_cleaned'], format='%m-%d-%Y', errors='coerce')

# THIS IS THE KEY FIX: Convert weather dates to date-only (remove time component)
weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
weather_df['date'] = pd.to_datetime(weather_df['date'])  # Convert back to datetime but without time

# Convert medical dates to date-only as well
medical_df['date_cleaned'] = medical_df['date_cleaned'].dt.date
medical_df['date_cleaned'] = pd.to_datetime(medical_df['date_cleaned'])

# Fix AQI dates the same way
if 'date' in aqi_df.columns:
    aqi_df['date'] = pd.to_datetime(aqi_df['date']).dt.date
    aqi_df['date'] = pd.to_datetime(aqi_df['date'])

print(f"\n✅ AFTER FIXES:")
print(f"Medical date range: {medical_df['date_cleaned'].min()} to {medical_df['date_cleaned'].max()}")
print(f"Weather date range: {weather_df['date'].min()} to {weather_df['date'].max()}")

# Test the fix
test_medical = medical_df[['date_cleaned']].drop_duplicates().head(5)
test_merge = test_medical.merge(weather_df, left_on='date_cleaned', right_on='date', how='left')
print(f"\n🧪 TEST MERGE:")
print(f"Non-null temperature values: {test_merge['temperature_2m_mean'].notna().sum()} out of {len(test_merge)}")
print("Sample merged data:")
print(test_merge[['date_cleaned', 'date', 'temperature_2m_mean', 'relative_humidity_2m_mean']].head())

# Check data overlap again
medical_dates = set(medical_df['date_cleaned'].dt.date)
weather_dates = set(weather_df['date'].dt.date)
overlap = medical_dates & weather_dates
print(f"\nOverlap check: {len(overlap)} overlapping dates")

# Add academic period mapping
def get_academic_period(date):
    if pd.isna(date):
        return 'unknown'
    month = date.month
    if month in [8, 9, 10]:
        return 'prelim'
    elif month in [11, 12, 1]:
        return 'midterm' 
    elif month in [2, 3, 4]:
        return 'finals'
    else:
        return 'break'

medical_df['academic_period'] = medical_df['date_cleaned'].apply(get_academic_period)
medical_df['is_exam_week'] = medical_df['date_cleaned'].apply(is_exam_week)

# Symptom categorization function
def categorize_symptoms(symptoms_str):
    # if pd.isna(symptoms_str) or symptoms_str == '':
    #     return {'respiratory': 0, 'digestive': 0, 'pain': 0, 'fever': 0, 'other': 0}
    
    # symptoms_lower = symptoms_str.lower()
    # respiratory_symptoms = ['cold', 'cough', 'asthma', 'runny nose', 'sore throat', 'itchy throat', 'auri']
    # digestive_symptoms = ['stomach ache', 'hyperacidity', 'lbm', 'diarrhea', 'vomiting', 'epigastric pain']
    # pain_symptoms = ['headache', 'body pain', 'muscle strain', 'chest pain', 'toothache', 'dysmenorrhea', 'cramps']
    # fever_symptoms = ['fever', 'malaise', 'infection']
    
    # counts = {
    #     'respiratory': sum(1 for sym in respiratory_symptoms if sym in symptoms_lower),
    #     'digestive': sum(1 for sym in digestive_symptoms if sym in symptoms_lower),
    #     'pain': sum(1 for sym in pain_symptoms if sym in symptoms_lower),
    #     'fever': sum(1 for sym in fever_symptoms if sym in symptoms_lower),
    #     'other': 0
    # }
    if pd.isna(symptoms_str) or symptoms_str == '':
        return {'respiratory': 0, 'digestive': 0, 'pain': 0, 'fever': 0, 
                'neurological': 0, 'dermatological': 0, 'cardiovascular': 0, 
                'urological': 0, 'other': 0}
    
    symptoms_lower = symptoms_str.lower()
    
    # Respiratory system
    respiratory_symptoms = [
        'cold', 'cough', 'asthma', 'runny nose', 'sore throat', 'itchy throat', 
        'auri', 'shortness of breath', 'hyperventilation', 'earache', 'nosebleed'
    ]
    
    # Digestive system
    digestive_symptoms = [
        'stomach ache', 'hyperacidity', 'lbm', 'diarrhea', 'vomiting', 
        'epigastric pain', 'dry mouth'
    ]
    
    # Pain and musculoskeletal
    pain_symptoms = [
        'headache', 'body pain', 'muscle strain', 'chest pain', 'toothache', 
        'dysmenorrhea', 'cramps', 'menstrual cramps', 'stiff neck', 'migraine'
    ]
    
    # Fever and infections
    fever_symptoms = [
        'fever', 'malaise', 'infection'
    ]
    
    # Neurological and psychological
    neurological_symptoms = [
        'dizziness', 'anxiety'
    ]
    
    # Dermatological and injuries
    dermatological_symptoms = [
        'allergy', 'skin allergy', 'abrasion', 'wound', 'punctured wound', 
        'cut', 'pimple', 'hematoma', 'stitches', 'sprain', 'clammy skin'
    ]
    
    # Cardiovascular
    cardiovascular_symptoms = [
        'hypertension'
    ]
    
    # Urological
    urological_symptoms = [
        'uti'
    ]
    
    # Count matches for each category
    counts = {
        'respiratory': sum(1 for sym in respiratory_symptoms if sym in symptoms_lower),
        'digestive': sum(1 for sym in digestive_symptoms if sym in symptoms_lower),
        'pain': sum(1 for sym in pain_symptoms if sym in symptoms_lower),
        'fever': sum(1 for sym in fever_symptoms if sym in symptoms_lower),
        'neurological': sum(1 for sym in neurological_symptoms if sym in symptoms_lower),
        'dermatological': sum(1 for sym in dermatological_symptoms if sym in symptoms_lower),
        'cardiovascular': sum(1 for sym in cardiovascular_symptoms if sym in symptoms_lower),
        'urological': sum(1 for sym in urological_symptoms if sym in symptoms_lower),
        'other': 0
    }
    
    # Calculate uncategorized symptoms
    total_categorized = sum(counts.values()) - counts['other']
    total_symptoms = len([s for s in symptoms_str.split(',') if s.strip()])
    counts['other'] = max(0, total_symptoms - total_categorized)
    
    return counts
    
    total_categorized = sum(counts.values()) - counts['other']
    total_symptoms = len([s for s in symptoms_str.split(',') if s.strip()])
    counts['other'] = max(0, total_symptoms - total_categorized)
    
    return counts

# Student vs staff analysis function
def analyze_courses(courses_list):
    student_courses = ['bslm', 'bsba', 'bscrim', 'bse', 'beed', 'baels', 'bscs', 'bstm', 'bat', 'bsit', 
                       'bsentrep', 'bshm', 'bsma', 'bsais', 'bapos', 'bsitelectech', 'bsed', 'bsitautotech', 
                       'ced', 'ccsict', 'bped', 'bsemc']
    
    if not courses_list:
        return {'student_visits': 0, 'staff_visits': 0, 'other_visits': 0}
    
    student_visits = sum(1 for course in courses_list if course in student_courses)
    staff_visits = sum(1 for course in courses_list if course == 'staff')
    other_visits = len(courses_list) - student_visits - staff_visits
    
    return {'student_visits': student_visits, 'staff_visits': staff_visits, 'other_visits': other_visits}

# ==========================================
# VERSION 1: DAILY EXCLUDING NO VISITS
# ==========================================

print("\n📊 Creating Daily Aggregates (Excluding No Visits)...")

# Aggregate medical data by day (only days with visits)
daily_medical_visits_only = medical_df.groupby(['date_cleaned', 'academic_period']).agg({
    'normalized_symptoms': lambda x: ','.join(x.dropna()),
    'course_mapped': lambda x: list(x.dropna()),
    'age': 'mean',
    'gender': lambda x: list(x.dropna())
}).reset_index()

# Add visit count
visit_counts = medical_df.groupby('date_cleaned').size().reset_index(name='visit_count')
daily_medical_visits_only = daily_medical_visits_only.merge(visit_counts, on='date_cleaned')

# Add symptom categories
symptom_categories = daily_medical_visits_only['normalized_symptoms'].apply(categorize_symptoms)
daily_medical_visits_only['respiratory_count'] = [cat['respiratory'] for cat in symptom_categories]
daily_medical_visits_only['digestive_count'] = [cat['digestive'] for cat in symptom_categories]
daily_medical_visits_only['pain_count'] = [cat['pain'] for cat in symptom_categories]
daily_medical_visits_only['fever_count'] = [cat['fever'] for cat in symptom_categories]
daily_medical_visits_only['neurological_count'] = [cat['neurological'] for cat in symptom_categories]  # NEW
daily_medical_visits_only['dermatological_count'] = [cat['dermatological'] for cat in symptom_categories]  # NEW
daily_medical_visits_only['cardiovascular_count'] = [cat['cardiovascular'] for cat in symptom_categories]  # NEW
daily_medical_visits_only['urological_count'] = [cat['urological'] for cat in symptom_categories]  # NEW
daily_medical_visits_only['other_count'] = [cat['other'] for cat in symptom_categories]

# Add course analysis
course_analysis = daily_medical_visits_only['course_mapped'].apply(analyze_courses)
daily_medical_visits_only['student_visits'] = [ca['student_visits'] for ca in course_analysis]
daily_medical_visits_only['staff_visits'] = [ca['staff_visits'] for ca in course_analysis]
daily_medical_visits_only['other_visits'] = [ca['other_visits'] for ca in course_analysis]

# ==========================================
# VERSION 2: DAILY INCLUDING NO VISITS
# ==========================================

print("📊 Creating Daily Aggregates (Including No Visits)...")

# Create complete date range
start_date = medical_df['date_cleaned'].min()
end_date = medical_df['date_cleaned'].max()
complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Create complete daily dataframe
complete_daily = pd.DataFrame({'date': complete_dates})
complete_daily['academic_period'] = complete_daily['date'].apply(get_academic_period)
complete_daily['is_exam_week'] = complete_daily['date'].apply(is_exam_week)

# Add temporal features
complete_daily['year'] = complete_daily['date'].dt.year
complete_daily['month'] = complete_daily['date'].dt.month
complete_daily['day_of_week'] = complete_daily['date'].dt.dayofweek
complete_daily['is_weekend'] = complete_daily['day_of_week'].isin([5, 6])
complete_daily['week_of_year'] = complete_daily['date'].dt.isocalendar().week

# Merge with visits-only data
daily_medical_complete = complete_daily.merge(daily_medical_visits_only, left_on='date', right_on='date_cleaned', how='left')

# Fill missing values with zeros/defaults
fill_columns = [
    'visit_count', 
    'respiratory_count', 'digestive_count', 'pain_count', 'fever_count', 
    'neurological_count', 'dermatological_count', 'cardiovascular_count', 'urological_count',
    'other_count',
    'student_visits', 'staff_visits', 'other_visits'
]
for col in fill_columns:
    daily_medical_complete[col] = daily_medical_complete[col].fillna(0)

daily_medical_complete['age'] = daily_medical_complete['age'].fillna(np.nan)
daily_medical_complete['normalized_symptoms'] = daily_medical_complete['normalized_symptoms'].fillna('')
daily_medical_complete['course_mapped'] = daily_medical_complete['course_mapped'].fillna('').apply(lambda x: [] if x == '' else x)
daily_medical_complete['gender'] = daily_medical_complete['gender'].fillna('').apply(lambda x: [] if x == '' else x)

# Use 'academic_period_x' from the complete dates (more reliable)
if 'academic_period_x' in daily_medical_complete.columns:
    daily_medical_complete['academic_period'] = daily_medical_complete['academic_period_x']
    daily_medical_complete = daily_medical_complete.drop(['academic_period_x', 'academic_period_y'], axis=1, errors='ignore')

# ==========================================
# ENVIRONMENTAL DATA MERGE
# ==========================================

print("🌤️ Merging Environmental Data...")

# Version 1: Visits only
print("Merging visits-only data...")
# Create weather data without date column for merging
weather_for_merge = weather_df.set_index('date')
daily_visits_only = daily_medical_visits_only.merge(weather_for_merge, left_on='date_cleaned', right_index=True, how='left')
print(f"✅ Weather merge: {len(daily_visits_only)} rows")

# Add AQI data
if len(aqi_df) > 0:
    aqi_for_merge = aqi_df.set_index('date')
    daily_visits_only = daily_visits_only.merge(aqi_for_merge, left_on='date_cleaned', right_index=True, how='left')
    print(f"✅ AQI merge: {len(daily_visits_only)} rows")

# Version 2: Complete days
print("Merging complete data...")
daily_complete = daily_medical_complete.merge(weather_for_merge, left_on='date', right_index=True, how='left')
print(f"✅ Weather merge: {len(daily_complete)} rows")

if len(aqi_df) > 0:
    daily_complete = daily_complete.merge(aqi_for_merge, left_on='date', right_index=True, how='left')
    print(f"✅ AQI merge: {len(daily_complete)} rows")

# ==========================================
# ADD DERIVED FEATURES
# ==========================================

def add_derived_features(df):
    """Add derived environmental features to dataframe"""
    print(f"Adding derived features...")
    
    # Temperature features
    if 'temperature_2m_max' in df.columns and 'temperature_2m_min' in df.columns:
        df['temperature_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
        print("✅ Added temperature_range")
    
    if 'temperature_2m_mean' in df.columns and 'relative_humidity_2m_mean' in df.columns:
        df['heat_humidity_index'] = df['temperature_2m_mean'] * (df['relative_humidity_2m_mean'] / 100)
        print("✅ Added heat_humidity_index")
    
    # Weather conditions
    if 'precipitation_sum' in df.columns:
        df['is_rainy_day'] = df['precipitation_sum'] > 5
        print("✅ Added is_rainy_day")
    if 'temperature_2m_max' in df.columns:
        df['is_hot_day'] = df['temperature_2m_max'] > 32
        print("✅ Added is_hot_day")
    if 'relative_humidity_2m_mean' in df.columns:
        df['is_humid_day'] = df['relative_humidity_2m_mean'] > 80
        print("✅ Added is_humid_day")
    
    # Air quality conditions  
    if 'pm2_5_mean' in df.columns:
        df['is_high_pollution'] = df['pm2_5_mean'] > 25
        print("✅ Added pollution indicators")
    
    # Fill missing environmental data
    env_columns = ['temperature_2m_mean', 'relative_humidity_2m_mean', 'pm2_5_mean', 'pm10_mean']
    for col in env_columns:
        if col in df.columns:
            before_fill = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled {col}: {before_fill} NaN → 0 NaN")
    
    return df

# Apply derived features
daily_visits_only = add_derived_features(daily_visits_only)
daily_complete = add_derived_features(daily_complete)

# Rename date columns for consistency
if 'date_x' in daily_visits_only.columns:
    daily_visits_only = daily_visits_only.rename(columns={'date_x': 'date'})

# ==========================================
# SAVE FILES AND SUMMARY
# ==========================================

# Save both versions
daily_visits_only.to_csv('medisense/backend/data/final/daily_visits_only.csv', index=False)
daily_complete.to_csv('medisense/backend/data/final/daily_complete.csv', index=False)

print(f"\n📊 DAILY AGGREGATION SUMMARY:")
print(f"=" * 50)
print(f"Daily (Visits Only):")
print(f"  - Total days: {len(daily_visits_only)}")
print(f"  - Average visits per day: {daily_visits_only['visit_count'].mean():.2f}")
print(f"  - Max visits per day: {daily_visits_only['visit_count'].max()}")

print(f"\nDaily (Complete):")
print(f"  - Total days: {len(daily_complete)}")
print(f"  - Days with visits: {(daily_complete['visit_count'] > 0).sum()}")
print(f"  - Zero-visit days: {(daily_complete['visit_count'] == 0).sum()}")
print(f"  - Average visits per day: {daily_complete['visit_count'].mean():.2f}")

print(f"\n💾 SAVED FILES:")
print(f"  - medisense/backend/data/final/daily_visits_only.csv")
print(f"  - medisense/backend/data/final/daily_complete.csv")

# Final data quality check
print(f"\n🔍 FINAL DATA QUALITY CHECK:")
temp_cols = [col for col in daily_complete.columns if 'temp' in col.lower()]
for col in temp_cols[:3]:
    if col in daily_complete.columns:
        nan_count = daily_complete[col].isna().sum()
        mean_val = daily_complete[col].mean()
        print(f"  {col}: {nan_count} NaN, mean = {mean_val:.2f}°C")

print(f"\n🎯 READY FOR XGBOOST!")

# Show sample of successful data
print(f"\n📋 SAMPLE DATA:")
sample_cols = ['date', 'academic_period', 'visit_count', 'respiratory_count', 'temperature_2m_mean', 'relative_humidity_2m_mean']
available_cols = [col for col in sample_cols if col in daily_complete.columns]
print(daily_complete[available_cols].head(10))