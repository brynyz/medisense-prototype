import pandas as pd
import numpy as np
from datetime import datetime

# Load your datasets
medical_df = pd.read_csv('medisense/backend/data/cleaned/cleaned_colummns.csv')
weather_df = pd.read_csv('medisense/backend/data/environment/historical_weather.csv')  
aqi_df = pd.read_csv('medisense/backend/data/environment/daily_aqi.csv')

# Fix date formats
medical_df['date_cleaned'] = pd.to_datetime(medical_df['date_cleaned'], format='%m-%d-%Y', errors='coerce')
weather_df['date'] = pd.to_datetime(weather_df['date'])
aqi_df['date'] = pd.to_datetime(aqi_df['date'])

# Create week identifiers
medical_df['year_week'] = medical_df['date_cleaned'].dt.strftime('%Y-W%U')
weather_df['year_week'] = weather_df['date'].dt.strftime('%Y-W%U')
aqi_df['year_week'] = aqi_df['date'].dt.strftime('%Y-W%U')

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

# Symptom categorization function
def categorize_symptoms(symptoms_str):
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
# VERSION 1: WEEKLY EXCLUDING NO VISITS
# ==========================================

print("Creating Weekly Aggregates (Excluding No Visits)...")

# Aggregate medical data by week (only weeks with visits)
weekly_medical_visits_only = medical_df.groupby(['year_week', 'academic_period']).agg({
    'normalized_symptoms': lambda x: ','.join(x.dropna()),
    'course_mapped': lambda x: list(x.dropna()),
    'age': 'mean',
    'gender': lambda x: list(x.dropna()),
    'date_cleaned': ['min', 'max', 'count']
}).reset_index()

# Flatten columns
weekly_medical_visits_only.columns = ['year_week', 'academic_period', 'symptoms_combined', 'courses_list', 
                                     'avg_age', 'genders_list', 'week_start', 'week_end', 'visit_count']

# Add symptom categories
symptom_categories = weekly_medical_visits_only['symptoms_combined'].apply(categorize_symptoms)
weekly_medical_visits_only['respiratory_count'] = [cat['respiratory'] for cat in symptom_categories]
weekly_medical_visits_only['digestive_count'] = [cat['digestive'] for cat in symptom_categories]
weekly_medical_visits_only['pain_count'] = [cat['pain'] for cat in symptom_categories]
weekly_medical_visits_only['fever_count'] = [cat['fever'] for cat in symptom_categories]
weekly_medical_visits_only['neurological_count'] = [cat['neurological'] for cat in symptom_categories]  # NEW
weekly_medical_visits_only['dermatological_count'] = [cat['dermatological'] for cat in symptom_categories]  # NEW
weekly_medical_visits_only['cardiovascular_count'] = [cat['cardiovascular'] for cat in symptom_categories]  # NEW
weekly_medical_visits_only['urological_count'] = [cat['urological'] for cat in symptom_categories]  # NEW
weekly_medical_visits_only['other_count'] = [cat['other'] for cat in symptom_categories]

# Add course analysis
course_analysis = weekly_medical_visits_only['courses_list'].apply(analyze_courses)
weekly_medical_visits_only['student_visits'] = [ca['student_visits'] for ca in course_analysis]
weekly_medical_visits_only['staff_visits'] = [ca['staff_visits'] for ca in course_analysis]
weekly_medical_visits_only['other_visits'] = [ca['other_visits'] for ca in course_analysis]

# ==========================================
# VERSION 2: WEEKLY INCLUDING NO VISITS
# ==========================================

print("Creating Weekly Aggregates (Including No Visits)...")

# Create complete date range for weeks
start_date = medical_df['date_cleaned'].min()
end_date = medical_df['date_cleaned'].max()
date_range = pd.date_range(start=start_date, end=end_date, freq='W')
all_weeks = [d.strftime('%Y-W%U') for d in date_range]

# Create complete weekly dataframe
complete_weekly = pd.DataFrame({'year_week': all_weeks})
complete_weekly['week_start_date'] = pd.to_datetime(
    [f"{week.split('-W')[0]}-W{week.split('-W')[1]}-1" for week in all_weeks],
    format='%Y-W%U-%w'
)
complete_weekly['academic_period'] = complete_weekly['week_start_date'].apply(get_academic_period)

# Merge with visits-only data
weekly_medical_complete = complete_weekly.merge(weekly_medical_visits_only, on=['year_week', 'academic_period'], how='left')

# Fill missing values with zeros/defaults
fill_columns = ['visit_count', 'respiratory_count', 'digestive_count', 'pain_count', 'fever_count', 
                'neurological_count', 'dermatological_count', 'cardiovascular_count', 'urological_count',
                'other_count', 'student_visits', 'staff_visits', 'other_visits']
for col in fill_columns:
    weekly_medical_complete[col] = weekly_medical_complete[col].fillna(0)

weekly_medical_complete['avg_age'] = weekly_medical_complete['avg_age'].fillna(np.nan)  # Default age
weekly_medical_complete['symptoms_combined'] = weekly_medical_complete['symptoms_combined'].fillna('')
weekly_medical_complete['courses_list'] = weekly_medical_complete['courses_list'].fillna('').apply(lambda x: [] if x == '' else x)
weekly_medical_complete['genders_list'] = weekly_medical_complete['genders_list'].fillna('').apply(lambda x: [] if x == '' else x)

# Set week start/end for zero-visit weeks
weekly_medical_complete['week_start'] = weekly_medical_complete['week_start'].fillna(weekly_medical_complete['week_start_date'])
weekly_medical_complete['week_end'] = weekly_medical_complete['week_end'].fillna(weekly_medical_complete['week_start_date'])

# ==========================================
# ENVIRONMENTAL DATA AGGREGATION
# ==========================================

print("Aggregating Environmental Data...")

# Weekly weather aggregation
weekly_weather = weather_df.groupby('year_week').agg({
    'temperature_2m_mean': 'mean',
    'temperature_2m_max': 'max',
    'temperature_2m_min': 'min',
    'relative_humidity_2m_mean': 'mean',
    'precipitation_sum': 'sum',
    'wind_speed_10m_mean': 'mean',
    'sunshine_duration': 'sum'
}).round(2).reset_index()

# Weekly AQI aggregation
weekly_aqi = aqi_df.groupby('year_week').agg({
    'pm2_5_mean': 'mean',
    'pm10_mean': 'mean',
    'ozone_mean': 'mean',
    'carbon_monoxide_mean': 'mean',
    'nitrogen_dioxide_mean': 'mean',
    'respiratory_risk_score': 'mean',
    'aqi_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
}).round(2).reset_index()

# ==========================================
# MERGE AND ADD DERIVED FEATURES
# ==========================================

def add_derived_features(df):
    """Add derived environmental features to dataframe"""
    # Temperature features
    df['temperature_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
    df['heat_humidity_index'] = df['temperature_2m_mean'] * (df['relative_humidity_2m_mean'] / 100)
    
    # Weather conditions
    df['is_rainy_week'] = df['precipitation_sum'] > 10
    df['is_hot_week'] = df['temperature_2m_max'] > 32
    df['is_humid_week'] = df['relative_humidity_2m_mean'] > 80
    
    # Air quality conditions
    df['is_high_pollution'] = df['pm2_5_mean'] > 25
    df['is_very_high_pollution'] = df['pm2_5_mean'] > 50
    
    # Combined risk indicators
    df['weather_stress_score'] = (
        df['is_hot_week'].astype(int) + 
        df['is_humid_week'].astype(int) + 
        df['is_rainy_week'].astype(int)
    )
    
    # Fill missing environmental data
    env_columns = ['temperature_2m_mean', 'relative_humidity_2m_mean', 'pm2_5_mean', 'pm10_mean']
    for col in env_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df

# Merge and process both versions
print("Merging datasets...")

# Version 1: Visits only
weekly_visits_only = weekly_medical_visits_only.merge(weekly_weather, on='year_week', how='left')
weekly_visits_only = weekly_visits_only.merge(weekly_aqi, on='year_week', how='left')
weekly_visits_only = add_derived_features(weekly_visits_only)

# Version 2: Complete weeks
weekly_complete = weekly_medical_complete.merge(weekly_weather, on='year_week', how='left')
weekly_complete = weekly_complete.merge(weekly_aqi, on='year_week', how='left')
weekly_complete = add_derived_features(weekly_complete)

# ==========================================
# SAVE FILES AND SUMMARY
# ==========================================

# Save both versions
weekly_visits_only.to_csv('medisense/backend/data/final/weekly_visits_only.csv', index=False)
weekly_complete.to_csv('medisense/backend/data/final/weekly_complete.csv', index=False)

print(f"\n📊 WEEKLY AGGREGATION SUMMARY:")
print(f"=" * 50)
print(f"Weekly (Visits Only):")
print(f"  - Total weeks: {len(weekly_visits_only)}")
print(f"  - Average visits per week: {weekly_visits_only['visit_count'].mean():.1f}")
print(f"  - Date range: {weekly_visits_only['week_start'].min()} to {weekly_visits_only['week_end'].max()}")

print(f"\nWeekly (Complete):")
print(f"  - Total weeks: {len(weekly_complete)}")
print(f"  - Weeks with visits: {(weekly_complete['visit_count'] > 0).sum()}")
print(f"  - Zero-visit weeks: {(weekly_complete['visit_count'] == 0).sum()}")
print(f"  - Average visits per week: {weekly_complete['visit_count'].mean():.1f}")

print(f"\nAcademic Period Distribution (Complete):")
print(weekly_complete['academic_period'].value_counts())

print(f"\n💾 SAVED FILES:")
print(f"  - medisense/backend/data/final/weekly_visits_only.csv")
print(f"  - medisense/backend/data/final/weekly_complete.csv")

print(f"\n🎯 READY FOR XGBOOST!")