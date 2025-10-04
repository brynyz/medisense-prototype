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

# Aggregate medical data by week
weekly_medical = medical_df.groupby(['year_week', 'academic_period']).agg({
    'normalized_symptoms': lambda x: ','.join(x.dropna()),  # Combine all symptoms
    'course_mapped': lambda x: list(x.dropna()),           # List of courses
    'age': 'mean',                                          # Average age
    'gender': lambda x: list(x.dropna()),                  # List of genders
    'date_cleaned': ['min', 'max', 'count']                 # Week start, end, visit count
}).reset_index()

# Flatten columns
weekly_medical.columns = ['year_week', 'academic_period', 'symptoms_combined', 'courses_list', 
                         'avg_age', 'genders_list', 'week_start', 'week_end', 'visit_count']

# Create symptom categories for weekly data
def categorize_weekly_symptoms(symptoms_str):
    if pd.isna(symptoms_str) or symptoms_str == '':
        return {'respiratory': 0, 'digestive': 0, 'pain': 0, 'other': 0}
    
    symptoms_lower = symptoms_str.lower()
    respiratory_symptoms = ['cold', 'cough', 'asthma', 'runny nose', 'sore throat', 'itchy throat', 'auri']
    digestive_symptoms = ['stomach ache', 'hyperacidity', 'lbm', 'diarrhea', 'vomiting', 'epigastric pain']
    pain_symptoms = ['headache', 'body pain', 'muscle strain', 'chest pain', 'toothache', 'dysmenorrhea', 'cramps']
    
    counts = {
        'respiratory': sum(1 for sym in respiratory_symptoms if sym in symptoms_lower),
        'digestive': sum(1 for sym in digestive_symptoms if sym in symptoms_lower),
        'pain': sum(1 for sym in pain_symptoms if sym in symptoms_lower),
        'other': 0  # Will calculate as remainder
    }
    
    total_categorized = counts['respiratory'] + counts['digestive'] + counts['pain']
    total_symptoms = len([s for s in symptoms_str.split(',') if s.strip()])
    counts['other'] = max(0, total_symptoms - total_categorized)
    
    return counts

# Apply symptom categorization
symptom_categories = weekly_medical['symptoms_combined'].apply(categorize_weekly_symptoms)
weekly_medical['respiratory_count'] = [cat['respiratory'] for cat in symptom_categories]
weekly_medical['digestive_count'] = [cat['digestive'] for cat in symptom_categories]
weekly_medical['pain_count'] = [cat['pain'] for cat in symptom_categories]
weekly_medical['other_count'] = [cat['other'] for cat in symptom_categories]

# Add student vs staff analysis
def analyze_weekly_courses(courses_list):
    student_courses = ['bslm', 'bsba', 'bscrim', 'bse', 'beed', 'baels', 'bscs', 'bstm', 'bat', 'bsit', 
                       'bsentrep', 'bshm', 'bsma', 'bsais', 'bapos', 'bsitelectech', 'bsed', 'bsitautotech', 
                       'ced', 'ccsict', 'bped', 'bsemc']
    
    student_visits = sum(1 for course in courses_list if course in student_courses)
    staff_visits = sum(1 for course in courses_list if course == 'staff')
    other_visits = len(courses_list) - student_visits - staff_visits
    
    return {'student_visits': student_visits, 'staff_visits': staff_visits, 'other_visits': other_visits}

course_analysis = weekly_medical['courses_list'].apply(analyze_weekly_courses)
weekly_medical['student_visits'] = [ca['student_visits'] for ca in course_analysis]
weekly_medical['staff_visits'] = [ca['staff_visits'] for ca in course_analysis]
weekly_medical['other_visits'] = [ca['other_visits'] for ca in course_analysis]

# Aggregate environmental data by week
weekly_weather = weather_df.groupby('year_week').agg({
    'temperature_2m_mean': 'mean',
    'temperature_2m_max': 'max',
    'temperature_2m_min': 'min',
    'relative_humidity_2m_mean': 'mean',
    'precipitation_sum': 'sum',
    'wind_speed_10m_mean': 'mean',
    'sunshine_duration': 'sum'
}).round(2).reset_index()

weekly_aqi = aqi_df.groupby('year_week').agg({
    'pm2_5_mean': 'mean',
    'pm10_mean': 'mean',
    'ozone_mean': 'mean',
    'carbon_monoxide_mean': 'mean',
    'nitrogen_dioxide_mean': 'mean',
    'respiratory_risk_score': 'mean',
    'aqi_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'  # Most common category
}).round(2).reset_index()

# Merge all datasets
final_weekly = weekly_medical.merge(weekly_weather, on='year_week', how='left')
final_weekly = final_weekly.merge(weekly_aqi, on='year_week', how='left')

# Add derived features
final_weekly['temperature_range'] = final_weekly['temperature_2m_max'] - final_weekly['temperature_2m_min']
final_weekly['heat_humidity_index'] = final_weekly['temperature_2m_mean'] * (final_weekly['relative_humidity_2m_mean'] / 100)
final_weekly['is_rainy_week'] = final_weekly['precipitation_sum'] > 10
final_weekly['is_hot_week'] = final_weekly['temperature_2m_max'] > 32
final_weekly['is_high_pollution'] = final_weekly['pm2_5_mean'] > 25

# Fill missing environmental data (some weeks might not have environmental data)
env_columns = ['temperature_2m_mean', 'relative_humidity_2m_mean', 'pm2_5_mean', 'pm10_mean']
for col in env_columns:
    if col in final_weekly.columns:
        final_weekly[col] = final_weekly[col].fillna(final_weekly[col].median())

print(f"\n✅ Weekly Analysis Summary:")
print(f"Total weeks: {len(final_weekly)}")
print(f"Weeks with medical visits: {(final_weekly['visit_count'] > 0).sum()}")
print(f"Average visits per week: {final_weekly['visit_count'].mean():.1f}")
print(f"Date range: {final_weekly['week_start'].min()} to {final_weekly['week_end'].max()}")

print(f"\nAcademic Period Distribution:")
print(final_weekly['academic_period'].value_counts())

print(f"\nSymptom Categories (weekly totals):")
print(f"Respiratory: {final_weekly['respiratory_count'].sum()}")
print(f"Digestive: {final_weekly['digestive_count'].sum()}")
print(f"Pain: {final_weekly['pain_count'].sum()}")
print(f"Other: {final_weekly['other_count'].sum()}")

# Save the final dataset
final_weekly.to_csv('medisense/backend/data/final/weekly_medical_environmental.csv', index=False)
print(f"\n💾 Saved: weekly_medical_environmental.csv")
print(f"Ready for XGBoost modeling! 🚀")

# Show sample data
print(f"\nSample data:")
print(final_weekly[['year_week', 'academic_period', 'visit_count', 'respiratory_count', 
                   'temperature_2m_mean', 'pm2_5_mean', 'is_rainy_week']].head(10))