"""
Individual Symptom Prediction Pipeline
=====================================
This script processes individual clinical logs (not daily aggregated) for 
dominant symptom prediction with proper class balancing and feature engineering.

Features:
- Individual symptom case analysis
- Class distribution testing and balancing
- Environmental and AQI data integration
- Comprehensive feature engineering
- Multiple balancing strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json

warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('../processed', exist_ok=True)
os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("INDIVIDUAL SYMPTOM PREDICTION PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n1. LOADING DATA...")
print("-" * 40)

# Load main clinical data
clinical_df = pd.read_csv('medisense/backend/data/cleaned/cleaned_colummns.csv')
print(f"Clinical data loaded: {clinical_df.shape[0]} records")

# Load environmental data
weather_df = pd.read_csv('medisense/backend/data/raw/historical_weather.csv')
print(f"Weather data loaded: {weather_df.shape[0]} days")

# Load AQI data
aqi_df = pd.read_csv('medisense/backend/data/raw/daily_aqi.csv')
print(f"AQI data loaded: {aqi_df.shape[0]} days")

# ============================================================================
# STEP 2: SYMPTOM CATEGORIZATION
# ============================================================================

print("\n2. CATEGORIZING SYMPTOMS...")
print("-" * 40)

def categorize_symptom(symptom_str):
    """
    Categorize individual symptoms into clinical categories.
    Returns the primary category for each symptom.
    """
    if pd.isna(symptom_str) or symptom_str == '':
        return 'unknown'
    
    symptom_lower = symptom_str.lower().strip()
    
    # Category definitions
    respiratory_symptoms = [
        'cold', 'cough', 'asthma', 'runny nose', 'sore throat', 'itchy throat', 
        'auri', 'shortness of breath', 'hyperventilation', 'earache', 'nosebleed',
        'hypertension'
    ]
    
    digestive_symptoms = [
        'stomach ache', 'hyperacidity', 'lbm', 'diarrhea', 'vomiting', 
        'epigastric pain', 'dry mouth'
    ]
    
    pain_musculoskeletal_symptoms = [
        'headache', 'body pain', 'muscle strain', 'chest pain', 'toothache', 
        'dysmenorrhea', 'cramps', 'menstrual cramps', 'stiff neck', 'migraine',
        'sprain'
    ]
    
    dermatological_trauma_symptoms = [
        'abrasion', 'wound', 'punctured wound', 'cut', 'pimple', 
        'hematoma', 'stitches'
    ]
    
    neuro_psych_symptoms = [
        'dizziness', 'anxiety'
    ]
    
    systemic_infectious_symptoms = [
        'fever', 'malaise', 'infection', 'uti', 'clammy skin', 'allergy', 
        'skin allergy'
    ]
    
    # Check each category
    for symptom in respiratory_symptoms:
        if symptom in symptom_lower:
            return 'respiratory'
    
    for symptom in digestive_symptoms:
        if symptom in symptom_lower:
            return 'digestive'
    
    for symptom in pain_musculoskeletal_symptoms:
        if symptom in symptom_lower:
            return 'pain_musculoskeletal'
    
    for symptom in dermatological_trauma_symptoms:
        if symptom in symptom_lower:
            return 'dermatological_trauma'
    
    for symptom in neuro_psych_symptoms:
        if symptom in symptom_lower:
            return 'neuro_psych'
    
    for symptom in systemic_infectious_symptoms:
        if symptom in symptom_lower:
            return 'systemic_infectious'
    
    for symptom in cardiovascular_chronic_symptoms:
        if symptom in symptom_lower:
            return 'cardiovascular_chronic'
    
    return 'other'

# Handle multiple symptoms per visit
expanded_records = []
for idx, row in clinical_df.iterrows():
    if pd.notna(row['normalized_symptoms']):
        symptoms = str(row['normalized_symptoms']).split(',')
        for symptom in symptoms:
            record = row.copy()
            record['individual_symptom'] = symptom.strip()
            record['symptom_category'] = categorize_symptom(symptom.strip())
            expanded_records.append(record)

# Create expanded dataframe
expanded_df = pd.DataFrame(expanded_records)
print(f"Expanded to {len(expanded_df)} individual symptom records")

# ============================================================================
# STEP 3: CLASS DISTRIBUTION ANALYSIS
# ============================================================================

print("\n3. ANALYZING CLASS DISTRIBUTION...")
print("-" * 40)

# Get initial distribution
initial_dist = expanded_df['symptom_category'].value_counts()
print("\nInitial Class Distribution:")
for category, count in initial_dist.items():
    percentage = (count / len(expanded_df)) * 100
    print(f"  {category:25s}: {count:5d} ({percentage:5.1f}%)")

# Visualize distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
initial_dist.plot(kind='bar', color='steelblue')
plt.title('Initial Symptom Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# ============================================================================
# STEP 4: CLASS BALANCING
# ============================================================================

print("\n4. BALANCING CLASSES...")
print("-" * 40)

# Strategy: Only balance classes with less than 30 samples, no downsampling
def balance_rare_classes(df, min_samples=30):
    """
    Balance only rare classes (< 30 samples) without downsampling.
    """
    category_counts = df['symptom_category'].value_counts()
    
    # Identify rare classes that need balancing
    rare_classes = category_counts[category_counts < min_samples].index.tolist()
    
    if rare_classes:
        print(f"Classes needing balancing (< {min_samples} samples): {rare_classes}")
        
        balanced_dfs = []
        for category in category_counts.index:
            category_df = df[df['symptom_category'] == category]
            current_size = len(category_df)
            
            if category in rare_classes:
                # Upsample rare classes to minimum threshold
                balanced_df = resample(category_df, 
                                     n_samples=min_samples,
                                     replace=True,
                                     random_state=42)
                print(f"  {category}: {current_size} -> {min_samples} (upsampled)")
            else:
                # Keep all samples for classes above threshold
                balanced_df = category_df
                print(f"  {category}: {current_size} (kept all samples)")
            
            balanced_dfs.append(balanced_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)
    else:
        print("No classes below minimum threshold. Keeping all data unchanged.")
        return df

# Apply balancing
balanced_df = balance_rare_classes(expanded_df.copy(), min_samples=30)

# Show balanced distribution
balanced_dist = balanced_df['symptom_category'].value_counts()
print("\nBalanced Class Distribution:")
for category, count in balanced_dist.items():
    percentage = (count / len(balanced_df)) * 100
    print(f"  {category:25s}: {count:5d} ({percentage:5.1f}%)")

# Visualize balanced distribution
plt.subplot(1, 2, 2)
balanced_dist.plot(kind='bar', color='coral')
plt.title('Balanced Symptom Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('medisense/backend/data/visualization/final/symptom_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 5: DATE PROCESSING AND MERGING
# ============================================================================

print("\n5. PROCESSING DATES AND MERGING DATA...")
print("-" * 40)

# Process dates in all dataframes with flexible parsing
# Handle different date formats in the cleaned data
def parse_date_flexible(date_str):
    """Parse dates with multiple possible formats."""
    if pd.isna(date_str):
        return pd.NaT
    
    # Try different formats
    formats = ['%m-%d-%Y', '%d-%m-%Y', '%m%d-%Y', '%Y-%m-%d']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    # If all formats fail, try pandas' intelligent parsing
    try:
        return pd.to_datetime(date_str)
    except:
        print(f"Could not parse date: {date_str}")
        return pd.NaT

balanced_df['date'] = balanced_df['date_cleaned'].apply(parse_date_flexible)

# Remove rows with invalid dates
invalid_dates = balanced_df['date'].isna()
if invalid_dates.sum() > 0:
    print(f"Removing {invalid_dates.sum()} rows with invalid dates")
    balanced_df = balanced_df[~invalid_dates].copy()

weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
aqi_df['date'] = pd.to_datetime(aqi_df['date'], format='%m/%d/%Y')

# Convert to date for merging
balanced_df['merge_date'] = balanced_df['date'].dt.date
weather_df['merge_date'] = weather_df['date']
aqi_df['merge_date'] = aqi_df['date'].dt.date

# Select relevant columns from weather
weather_cols = ['merge_date', 'temperature_2m_mean', 'relative_humidity_2m_mean', 
                'precipitation_sum', 'wind_speed_10m_mean']
weather_subset = weather_df[weather_cols].copy()
weather_subset.columns = ['merge_date', 'temp', 'humidity', 'rainfall', 'wind_speed']

# Select relevant columns from AQI
aqi_cols = ['merge_date', 'pm2_5_mean', 'pm10_mean', 'ozone_mean', 
            'respiratory_risk_score', 'aqi_category']
aqi_subset = aqi_df[aqi_cols].copy()
aqi_subset.columns = ['merge_date', 'pm2_5', 'pm10', 'ozone', 
                      'respiratory_risk', 'aqi_category']

# Merge environmental data
merged_df = balanced_df.merge(weather_subset, on='merge_date', how='left')
merged_df = merged_df.merge(aqi_subset, on='merge_date', how='left')

print(f"Merged dataset: {len(merged_df)} records")
print(f"Missing weather data: {merged_df['temp'].isna().sum()} records")
print(f"Missing AQI data: {merged_df['pm2_5'].isna().sum()} records")

# ============================================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================================

print("\n6. ENGINEERING FEATURES...")
print("-" * 40)

# Temporal features
merged_df['dow'] = merged_df['date'].dt.dayofweek
merged_df['month'] = merged_df['date'].dt.month
merged_df['week_of_year'] = merged_df['date'].dt.isocalendar().week
merged_df['is_weekend'] = (merged_df['dow'] >= 5).astype(int)

# Academic period features - Refactored for actual semester structure
def get_academic_period(date):
    """
    Determine academic period based on date.
    First semester: Starts 2nd week of August (around Aug 8-14)
    Second semester: Starts 2nd week of January (around Jan 8-14)
    Each semester has 18 weeks divided into:
    - Prelim: Weeks 1-6
    - Midterm: Weeks 7-12
    - Finals: Weeks 13-18
    """
    month = date.month
    day = date.day
    
    # Summer break (May to early August)
    if month in [5, 6, 7]:
        return 'summer_break'
    elif month == 8 and day < 8:
        return 'summer_break'
    
    # First Semester (August - December)
    elif month == 8:
        # Semester starts around Aug 8-14
        if day >= 8:
            weeks_since_start = ((day - 8) // 7) + 1
            if weeks_since_start <= 6:
                return 'first_sem_prelim'
            else:
                return 'first_sem_prelim'
    elif month == 9:
        # September: weeks 4-8 of first sem
        day_of_month = day
        weeks_since_aug8 = ((31 - 8) // 7) + (day_of_month // 7) + 1
        if weeks_since_aug8 <= 6:
            return 'first_sem_prelim'
        elif weeks_since_aug8 <= 12:
            return 'first_sem_midterm'
        else:
            return 'first_sem_midterm'
    elif month == 10:
        # October: weeks 8-12 of first sem (mostly midterm)
        return 'first_sem_midterm'
    elif month == 11:
        # November: weeks 13-16 of first sem (finals period)
        if day <= 20:
            return 'first_sem_finals'
        else:
            return 'first_sem_finals'
    elif month == 12:
        # December: end of first sem and Christmas break
        if day <= 15:
            return 'first_sem_finals'
        else:
            return 'christmas_break'
    
    # Christmas/New Year break
    elif month == 1 and day < 8:
        return 'christmas_break'
    
    # Second Semester (January - April)
    elif month == 1:
        # Semester starts around Jan 8-14
        if day >= 8:
            weeks_since_start = ((day - 8) // 7) + 1
            if weeks_since_start <= 6:
                return 'second_sem_prelim'
            else:
                return 'second_sem_prelim'
    elif month == 2:
        # February: weeks 4-8 of second sem
        day_of_month = day
        weeks_since_jan8 = ((31 - 8) // 7) + (day_of_month // 7) + 1
        if weeks_since_jan8 <= 6:
            return 'second_sem_prelim'
        elif weeks_since_jan8 <= 12:
            return 'second_sem_midterm'
        else:
            return 'second_sem_midterm'
    elif month == 3:
        # March: weeks 8-12 of second sem (mostly midterm)
        return 'second_sem_midterm'
    elif month == 4:
        # April: weeks 13-18 of second sem (finals period)
        if day <= 25:
            return 'second_sem_finals'
        else:
            return 'summer_break'
    else:
        return 'break'

merged_df['academic_period'] = merged_df['date'].apply(get_academic_period)

# Visit context features (daily aggregates)
daily_visits = merged_df.groupby('merge_date').size().reset_index(name='daily_visits_count')
merged_df = merged_df.merge(daily_visits, on='merge_date', how='left')

# Create lag features for visit patterns
visit_history = daily_visits.copy()
visit_history['visit_lag1'] = visit_history['daily_visits_count'].shift(1)
visit_history['visit_lag3'] = visit_history['daily_visits_count'].shift(3)
visit_history['visit_lag7'] = visit_history['daily_visits_count'].shift(7)
visit_history['visit_rollmean7'] = visit_history['daily_visits_count'].rolling(7, min_periods=1).mean()

# Merge lag features
merged_df = merged_df.merge(visit_history[['merge_date', 'visit_lag1', 'visit_lag3', 
                                           'visit_lag7', 'visit_rollmean7']], 
                           on='merge_date', how='left')

# Patient context features
# Handle missing ages
merged_df['age'].fillna(merged_df['age'].median(), inplace=True)

# Create age groups
merged_df['age_group'] = pd.cut(merged_df['age'], 
                                bins=[0, 18, 20, 22, 100],
                                labels=['<18', '18-20', '21-22', '>22'])

# Label encode gender (before course grouping)
from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
merged_df['gender_encoded'] = gender_encoder.fit_transform(merged_df['gender'].fillna('Unknown'))

# Course grouping (academic programs)
course_groups = {
    # Staff & Others
    "staff": "staff",
    "others": "oth",

    # College of Laws
    "bslm": "law",            # Bachelor of Science in Legal Management
    "jurisdoctor": "law",     # Juris Doctor (Law degree)

    # College of Business Management
    "bsba": "cbm",
    "bstm": "cbm",
    "bsentrep": "cbm",
    "bshm": "cbm",
    "bsma": "cbm",
    "bsais": "cbm",

    # College of Criminal Justice Education
    "bscrim": "ccje",
    "bslea": "ccje",          # Law Enforcement Administration (missing)

    # College of Education
    "bse": "ced",
    "beed": "ced",
    "bsed": "ced",
    "bped": "ced",
    "ced": "ced",
    "btved": "ced",           # Technical-Vocational Teacher Education (missing)
    "btle": "ced",            # Technology and Livelihood Education (missing)

    # College of Computing Studies, ICT
    "bscs": "ccsict",
    "bsit": "ccsict",
    "bsemc": "sas",           # Entertainment and Multi Media Computing (reclassified to SAS)
    "bsis": "ccsict",         # Information Systems (missing)
    "ccsict": "ccsict",

    # School of Arts & Sciences
    "baels": "sas",
    "bapos": "sas",
    "bacomm": "sas",          # Communication (missing)
    "bsbio": "sas",           # Biology (missing)
    "bsmath": "sas",          # Mathematics (missing)
    "bschem": "sas",          # Chemistry (missing)
    "bspsych": "sas",         # Psychology (missing)

    # Polytechnic School
    "bsitelectech": "poly",
    "bsitautotech": "poly",
    "bsindtech": "poly",      # Industrial Technology (missing)
    "mechanicaltech": "poly", # Mechanical Technology (missing)
    "refrigaircondtech": "poly", # Refrigeration & Airconditioning (missing)
    "assocaircraftmaint": "poly", # Aircraft Maintenance (missing)

    # Agriculture
    "bat": "agriculture",
    "bsagri": "agri",         # Agriculture major (missing)
    "bsagribiz": "agri",      # Agribusiness (missing)
    "bsenvi": "agri",         # Environmental Science (missing)
    "bsfor": "agri",          # Forestry (missing)
    "bsfisheries": "agri",    # Fisheries and Aquatic Sciences (missing)

    # Graduate Programs
    "dit": "grad",            # Doctor of Information Technology
    "mit": "grad",            # Master in Information Technology
    "ddsa": "grad",           # Diploma in Data Science Analytics
    "mba": "grad",            # Master of Business Administration (Extension)
    "mpa": "grad",            # Master in Public Administration (Extension)
    "masterlaws": "grad",     # Master of Laws (Consortium)
    "maed": "grad",           # MA in Education
    "phd_ed": "grad",         # PhD in Education
    "phd_animal": "grad",     # PhD in Animal Science
    "phd_crop": "grad",       # PhD in Crop Science
}
merged_df['course_group'] = merged_df['course_mapped'].map(course_groups).fillna('other')

# Label encode course_group
course_encoder = LabelEncoder()
merged_df['course_group_encoded'] = course_encoder.fit_transform(merged_df['course_group'])

# Environmental interaction features
merged_df['temp_humidity_index'] = merged_df['temp'] * merged_df['humidity'] / 100
merged_df['weather_stress'] = (
    (merged_df['temp'] > 30).astype(int) + 
    (merged_df['humidity'] > 80).astype(int) +
    (merged_df['rainfall'] > 10).astype(int)
)

# Fill missing environmental data with medians
env_features = ['temp', 'humidity', 'rainfall', 'wind_speed', 'pm2_5', 'pm10', 'ozone']
for feature in env_features:
    if feature in merged_df.columns:
        merged_df[feature].fillna(merged_df[feature].median(), inplace=True)

# Handle missing categorical data
merged_df['aqi_category'].fillna('unknown', inplace=True)
merged_df['respiratory_risk'].fillna(0, inplace=True)

print("Feature engineering completed")
print(f"Total features created: {len(merged_df.columns)}")

# ============================================================================
# STEP 7: PREPARE FINAL DATASET
# ============================================================================

print("\n7. PREPARING FINAL DATASET...")
print("-" * 40)

# Select final features
feature_columns = [
    # Temporal
    'dow', 'month', 'week_of_year', 'is_weekend',
    
    # Environmental
    'temp', 'humidity', 'rainfall', 'wind_speed',
    'pm2_5', 'pm10', 'ozone', 'respiratory_risk',
    'temp_humidity_index', 'weather_stress',
    
    # Visit Context
    'daily_visits_count', 'visit_lag1', 'visit_lag3', 
    'visit_lag7', 'visit_rollmean7',
    
    # Patient Context
    'age', 'gender_encoded', 'course_group_encoded',
    
    # Target
    'symptom_category'
]

# Add categorical features that still need encoding
categorical_features = ['academic_period', 'age_group', 'aqi_category']

# Create final dataset
final_df = merged_df[feature_columns + categorical_features].copy()

# One-hot encode remaining categorical features
final_df = pd.get_dummies(final_df, columns=categorical_features, drop_first=True)

print(f"Final dataset shape: {final_df.shape}")
print(f"Features: {final_df.shape[1] - 1}")
print(f"Samples: {final_df.shape[0]}")

# ============================================================================
# STEP 8: SAVE DATASETS
# ============================================================================

print("\n8. SAVING DATASETS...")
print("-" * 40)

# Handle any remaining NaN values before encoding and saving
print("Handling missing values...")
# Fill NaN values in numerical columns with median
numerical_cols = final_df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if final_df[col].isna().any():
        final_df[col].fillna(final_df[col].median(), inplace=True)
        print(f"  Filled {col} with median")

# Fill NaN values in categorical columns with mode or 'unknown'
categorical_cols = final_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'symptom_category':  # Don't modify target
        if final_df[col].isna().any():
            final_df[col].fillna('unknown', inplace=True)
            print(f"  Filled {col} with 'unknown'")

# Label encode the symptom category target before saving
symptom_encoder = LabelEncoder()
final_df['symptom_category_encoded'] = symptom_encoder.fit_transform(final_df['symptom_category'])

# Final check for NaN values
if final_df.isna().any().any():
    print("Warning: Dataset still contains NaN values!")
    print(final_df.isna().sum()[final_df.isna().sum() > 0])
else:
    print("All missing values handled successfully")

# Save main dataset with both encoded and original symptom categories
output_path = 'medisense/backend/data/final/symptom/individual_symptom_dataset.csv'
final_df.to_csv(output_path, index=False)
print(f"Main dataset saved to: {output_path}")

# Create train/test split
X = final_df.drop(['symptom_category', 'symptom_category_encoded'], axis=1)
y_encoded = final_df['symptom_category_encoded']

# Create DataFrame with encoded target for splitting
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Also keep the original labels for reference
y_train_original = symptom_encoder.inverse_transform(y_train_encoded)
y_test_original = symptom_encoder.inverse_transform(y_test_encoded)

# Save train/test sets with encoded targets
train_df = pd.concat([X_train, pd.Series(y_train_encoded, index=X_train.index, name='symptom_category_encoded')], axis=1)
test_df = pd.concat([X_test, pd.Series(y_test_encoded, index=X_test.index, name='symptom_category_encoded')], axis=1)

# Also add original labels for reference
train_df['symptom_category_original'] = y_train_original
test_df['symptom_category_original'] = y_test_original

# Final check for NaN values in train/test sets
print("\nChecking for NaN values in train/test sets...")
if train_df.isna().any().any():
    print("Warning: Training set contains NaN values!")
    print(train_df.isna().sum()[train_df.isna().sum() > 0])
else:
    print("Training set: No NaN values found")
    
if test_df.isna().any().any():
    print("Warning: Test set contains NaN values!")
    print(test_df.isna().sum()[test_df.isna().sum() > 0])
else:
    print("Test set: No NaN values found")

train_df.to_csv('medisense/backend/data/final/symptom/symptom_train.csv', index=False)
test_df.to_csv('medisense/backend/data/final/symptom/symptom_test.csv', index=False)

print(f"\nTraining set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")
print(f"Symptom categories: {list(symptom_encoder.classes_)}")

# Save encoders for later use
joblib.dump(gender_encoder, 'medisense/backend/models/gender_encoder.pkl')
joblib.dump(course_encoder, 'medisense/backend/models/course_encoder.pkl')
joblib.dump(symptom_encoder, 'medisense/backend/models/symptom_encoder.pkl')
print("Encoders saved to medisense/backend/models/")

# ============================================================================
# STEP 9: DATASET SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print("\nFinal Class Distribution:")
final_dist = final_df['symptom_category'].value_counts()
for category, count in final_dist.items():
    percentage = (count / len(final_df)) * 100
    print(f"  {category:25s}: {count:5d} ({percentage:5.1f}%)")

print("\nFeature Categories:")
print(f"  Temporal features: 4")
print(f"  Environmental features: 10")
print(f"  Visit context features: 5")
print(f"  Patient features: 1 + encoded categoricals")
print(f"  Total features: {X.shape[1]}")

print("\nDataset Statistics:")
print(f"  Total samples: {len(final_df)}")
print(f"  Training samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Number of classes: {len(final_dist)}")
print(f"  Class balance ratio: {final_dist.max() / final_dist.min():.2f}:1")

# Create summary report
summary = {
    'total_samples': len(final_df),
    'num_features': X.shape[1],
    'num_classes': len(final_dist),
    'class_distribution': final_dist.to_dict(),
    'train_size': len(train_df),
    'test_size': len(test_df),
    'features': list(X.columns)
}

# Save summary
with open('medisense/backend/data/final/symptom/symptom_dataset_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nNext steps:")
print("1. Review class distribution in symptom_class_distribution.png")
print("2. Train models using symptom_train.csv")
print("3. Evaluate on symptom_test.csv")
print("4. Consider further feature engineering if needed")
