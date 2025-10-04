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
    """
    Map dates to academic periods based on university calendar:
    - 2 semesters per year (Aug-Dec, Jan-May)
    - 3 periods per semester (Prelim, Midterms, Finals)
    - Each period is 6 weeks, with week 6 being exam week
    """
    if pd.isna(date):
        return 'unknown'
    
    month = date.month
    day = date.day
    
    # First Semester (August 1 - December)
    if month == 8:
        return 'prelim'  # Aug 1 - Sep 11 (6 weeks)
    elif month == 9:
        if day <= 11:
            return 'prelim'  # Up to week 6 (exam week ~Sep 5-11)
        else:
            return 'midterm'  # Midterm starts ~Sep 12
    elif month == 10:
        if day <= 23:
            return 'midterm'  # Sep 12 - Oct 23 (6 weeks)
        else:
            return 'finals'  # Finals starts ~Oct 24
    elif month == 11:
        return 'finals'  # Oct 24 - Dec 4 (6 weeks)
    elif month == 12:
        if day <= 4:
            return 'finals'  # Finals exam week ~Nov 28-Dec 4
        else:
            return 'break'  # Christmas break
    
    # Second Semester (January 9 - May)
    elif month == 1:
        if day < 9:
            return 'break'  # New Year break
        else:
            return 'prelim'  # Second sem prelim starts Jan 9
    elif month == 2:
        if day <= 19:
            return 'prelim'  # Jan 9 - Feb 19 (6 weeks)
        else:
            return 'midterm'  # Midterm starts ~Feb 20
    elif month == 3:
        return 'midterm'  # Feb 20 - Apr 2 (6 weeks)
    elif month == 4:
        if day <= 2:
            return 'midterm'  # Midterm exam week ~Mar 27-Apr 2
        else:
            return 'finals'  # Finals starts ~Apr 3
    elif month == 5:
        if day <= 14:
            return 'finals'  # Apr 3 - May 14 (6 weeks)
        else:
            return 'break'  # Summer break starts ~May 15
    elif month in [6, 7]:
        return 'break'  # Summer break
    else:
        return 'unknown'

def is_exam_week(date):
    """
    Determine if the date falls in an exam week (week 6 of each period)
    """
    if pd.isna(date):
        return False
    
    month = date.month
    day = date.day
    
    # First Semester exam weeks
    if (month == 9 and 5 <= day <= 11):  # Prelim exam week 6
        return True
    elif (month == 10 and 17 <= day <= 23):  # Midterm exam week 6
        return True
    elif (month == 11 and day >= 28) or (month == 12 and day <= 4):  # Finals exam week 6
        return True
    
    # Second Semester exam weeks
    elif (month == 2 and 13 <= day <= 19):  # Prelim exam week 6
        return True
    elif (month == 3 and day >= 27) or (month == 4 and day <= 2):  # Midterm exam week 6
        return True
    elif (month == 5 and 8 <= day <= 14):  # Finals exam week 6
        return True
    
    return False

medical_df['academic_period'] = medical_df['date_cleaned'].apply(get_academic_period)
medical_df['is_exam_week'] = medical_df['date_cleaned'].apply(is_exam_week)

# Symptom categorization function
def categorize_symptoms(symptoms_str):
    if pd.isna(symptoms_str) or symptoms_str == '':
        return {'respiratory': 0, 'digestive': 0, 'pain_musculoskeletal': 0, 
                'dermatological_trauma': 0, 'neuro_psych': 0, 'systemic_infectious': 0, 
                'cardiovascular_chronic': 0, 'other': 0}
    
    symptoms_lower = symptoms_str.lower()
    
    # 1. Respiratory System (Largely unchanged)
    respiratory_symptoms = [
        'cold', 'cough', 'asthma', 'runny nose', 'sore throat', 'itchy throat', 
        'auri', 'shortness of breath', 'hyperventilation', 'earache', 'nosebleed'
    ]
    
    # 2. Digestive System (Unchanged)
    digestive_symptoms = [
        'stomach ache', 'hyperacidity', 'lbm', 'diarrhea', 'vomiting', 
        'epigastric pain', 'dry mouth'
    ]
    
    # 3. Pain & Musculoskeletal (Refined)
    pain_musculoskeletal_symptoms = [
        'headache', 'body pain', 'muscle strain', 'chest pain', 'toothache', 
        'dysmenorrhea', 'cramps', 'menstrual cramps', 'stiff neck', 'migraine',
        'sprain'  # Moved here from 'Dermatological'
    ]
    
    # 4. Dermatological & Trauma (Refined)
    dermatological_trauma_symptoms = [
        'abrasion', 'wound', 'punctured wound', 'cut', 'pimple', 
        'hematoma', 'stitches'
    ]
    
    # 5. Neurological & Psychological (Unchanged)
    neuro_psych_symptoms = [
        'dizziness', 'anxiety'
    ]
    
    # 6. Systemic & Infectious (New consolidated category)
    systemic_infectious_symptoms = [
        'fever', 'malaise', 'infection', 'uti', 'clammy skin', 'allergy', 
        'skin allergy'
    ]
    
    # 7. Cardiovascular / Chronic (New specific category)
    cardiovascular_chronic_symptoms = [
        'hypertension'
    ]
    
    # Count matches for each category
    counts = {
        'respiratory': sum(1 for sym in respiratory_symptoms if sym in symptoms_lower),
        'digestive': sum(1 for sym in digestive_symptoms if sym in symptoms_lower),
        'pain_musculoskeletal': sum(1 for sym in pain_musculoskeletal_symptoms if sym in symptoms_lower),
        'dermatological_trauma': sum(1 for sym in dermatological_trauma_symptoms if sym in symptoms_lower),
        'neuro_psych': sum(1 for sym in neuro_psych_symptoms if sym in symptoms_lower),
        'systemic_infectious': sum(1 for sym in systemic_infectious_symptoms if sym in symptoms_lower),
        'cardiovascular_chronic': sum(1 for sym in cardiovascular_chronic_symptoms if sym in symptoms_lower),
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
daily_medical_visits_only['pain_musculoskeletal_count'] = [cat['pain_musculoskeletal'] for cat in symptom_categories]
daily_medical_visits_only['dermatological_trauma_count'] = [cat['dermatological_trauma'] for cat in symptom_categories]
daily_medical_visits_only['neuro_psych_count'] = [cat['neuro_psych'] for cat in symptom_categories]
daily_medical_visits_only['systemic_infectious_count'] = [cat['systemic_infectious'] for cat in symptom_categories]
daily_medical_visits_only['cardiovascular_chronic_count'] = [cat['cardiovascular_chronic'] for cat in symptom_categories]
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
    'respiratory_count', 'digestive_count', 'pain_musculoskeletal_count', 
    'dermatological_trauma_count', 'neuro_psych_count', 'systemic_infectious_count', 
    'cardiovascular_chronic_count', 'other_count',
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
print(daily_complete[available_cols].head())
