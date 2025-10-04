import pandas as pd
import re
from rapidfuzz import process, fuzz
import numpy as np
from datetime import datetime

df = pd.read_csv('medisense\\backend\\data\\cleaned\\cleaned.csv')

def clean_date_string(s):
    s = s.strip().lower()

    # Replace separators with "-"
    s = re.sub(r"[/.]", "-", s)

    # Replace month names with numbers
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "sept": "09", "oct": "10", "nov": "11", "dec": "12"
    }
    for k, v in months.items():
        s = re.sub(rf"{k}[a-z]*", v, s)

    # Remove extra text/commas
    s = re.sub(r"[^\d\-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")

    return s

df['course_cleaned'] = (
    df['course']
    .str.lower()
    .str.replace(r'[^\w\s,]', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace(r'\b\d+\b', '', regex=True)
    .str.replace(r',+', ',', regex=True)
    .str.strip(' ,')
)

# Canonical course list
canonical_course = [
    "staff", "bslm", "bsba", "bscrim", "bse", "beed", "baels", "bscs", "bstm", "bat", "bsit",
    "bsentrep", "bshm", "bsma", "bsais", "bapos", "bsitelectech", "bsed", "bsitautotech",
    "ced", "others", "ccsict", "bped", "bsemc"
]

# Manual mapping
manual_mapping = {
    "bsba": "bsba",
    "bscrim": "bscrim",
    "bse": "bse",
    "beed": "beed",
    "baels": "baels",
    "bscs": "bscs",
    "bstm": "bstm",
    "bat": "bat",
    "bsit": "bsit",
    "bsentrep": "bsentrep",
    "bsent": "bsentrep",
    "bshm": "bshm",
    "bsma": "bsma",
    "bsam": "bsma",
    "bsa": "bsba",
    "bas": "bsba",
    "bsais": "bsais",
    "bapos": "bapos",
    "bapod": "bapos",
    "bsitelectech": "bsitelectech",
    "bsitechelx": "bsitelectech",
    "bsitecelc": "bsitelectech",
    "bsauto": "bsitautotech",
    "entrep": "bsentrep",
    "bsesci": "bse",
    "bsed": "bse",
    "bse sci": "bse",
    "ced": "ced",
    "accounting": "bsba",
    "accountancy": "bsba",
    "bsc": "bscs",
    "bsitechauto": "bsitautotech",
    "bsitech auto": "bsitautotech",
    "bped": "bped",
    "bsemc": "bsemc",
    # Staff and others...
    "hr management": "staff",
    "hr": "staff",
    "osas": "staff",
    "security": "staff",
    "cashier": "staff",
    "scholarship": "staff",
    "clinic": "staff",
    "library": "staff",
    "extension": "staff",
    "registrar": "staff",
    "qa": "staff",
    "mis": "staff",
    "afs": "staff",
    "cbao": "staff",
    "gso": "staff",
    "faculty": "staff",
    "guidance": "staff",
    "sas": "staff",
    "cbm": "staff",
    "cbmhm": "staff",
    "planning": "staff",
    "procurement": "staff",
    "pd": "staff",
    "admin": "staff",
    "cso": "staff",
    "clinic": "staff",
    "eo sec": "staff",
    "eosec": "staff",
    "extension": "staff",
    "fic": "staff",
    "supply": "staff",
    "cashier": "staff",
    "rgo": "staff",
    # Others
    "nan": None,
    "none": None,
    "n/a": None,
    "": None,
    " ": None,
    "unknown": None,
    "ps": "others",
    "abe": "others",
    "iat": "others",
    "planning": "others",
    "rsd": "others",
    "aa": "others",
    "mfs": "others",
    "scholarship": "others",
    "bac": "others",
}

# Fuzzy mapping
def map_course(raw_course, manual_map, canonical_list, threshold=80):
    if pd.isna(raw_course) or raw_course.strip() == "":
        return "others"  # Default to "others" instead of None
    raw_course = raw_course.strip()
    if raw_course in manual_map:
        return manual_map[raw_course] if manual_map[raw_course] is not None else "others"
    match, score, _ = process.extractOne(raw_course, canonical_list, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return match
    return "others"  # Default to "others" for unmatched

df['course_mapped'] = df['course_cleaned'].apply(
    lambda x: map_course(x, manual_mapping, canonical_course)
)

# Print unmatched courses before mapping
unmatched_courses = set()
for course in df['course_cleaned'].unique():
    if pd.notna(course) and course.strip() != "":
        mapped = map_course(course, manual_mapping, canonical_course)
        if mapped == "others" and course not in manual_mapping:
            unmatched_courses.add(course)

print("\nCourses mapped to 'others' (originally unmatched):")
for course in sorted(unmatched_courses):
    print("-", course)

# Clean date strings
df['date_cleaned'] = df['date'].apply(clean_date_string)

# Advanced date cleaning and imputation
def smart_date_cleaning(date_str, row_index=None):
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Fix specific known issues
    if date_str.startswith('0417'):
        date_str = '04-17-2023'
    elif date_str.startswith('7-02'):
        date_str = '07-02-2025'
    
    # Try multiple date formats
    formats_to_try = ['%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
    
    for fmt in formats_to_try:
        try:
            parsed_date = pd.to_datetime(date_str, format=fmt)
            # Sanity check: reasonable date range
            if 2020 <= parsed_date.year <= 2025:
                return parsed_date
        except:
            continue
    
    # If all formats fail, try pandas general parser
    try:
        parsed_date = pd.to_datetime(date_str)
        if 2020 <= parsed_date.year <= 2025:
            return parsed_date
    except:
        pass
    
    return None  # Mark for imputation

df['date_cleaned'] = df.reset_index().apply(
    lambda row: smart_date_cleaning(row['date_cleaned'], row['index']), axis=1
)

# Date imputation strategy
def impute_missing_dates(df):
    # Sort by index to maintain chronological order
    df_sorted = df.sort_index()
    
    # For missing dates, use interpolation based on surrounding valid dates
    valid_dates = df_sorted['date_cleaned'].dropna()
    
    if len(valid_dates) > 0:
        # Fill missing dates with forward fill, then backward fill
        df_sorted['date_cleaned'] = df_sorted['date_cleaned'].fillna(method='ffill').fillna(method='bfill')
        
        # If still missing, use median date
        if df_sorted['date_cleaned'].isna().any():
            median_date = valid_dates.median()
            df_sorted['date_cleaned'] = df_sorted['date_cleaned'].fillna(median_date)
    
    return df_sorted['date_cleaned']

df['date_cleaned'] = impute_missing_dates(df)

# Clean gender with smart defaults
def clean_gender(gender_val):
    if pd.isna(gender_val):
        return "Unknown"
    
    gender_str = str(gender_val).strip().lower()
    
    if gender_str in ['m', 'male', 'man', 'boy']:
        return 'Male'
    elif gender_str in ['f', 'female', 'woman', 'girl']:
        return 'Female'
    else:
        return 'Unknown'  # Instead of dropping

df['gender'] = df['gender'].apply(clean_gender)

# Smart age imputation
def impute_ages(df):
    # Calculate rolling median age by course and time period
    df['year_month'] = pd.to_datetime(df['date_cleaned']).dt.to_period('M')
    
    # Group by course and calculate median age
    course_age_medians = df.groupby('course_mapped')['age'].median()
    
    # Global median as fallback
    global_median_age = df['age'].median()
    if pd.isna(global_median_age):
        global_median_age = 20  # Reasonable default for student population
    
    def fill_age(row):
        if pd.notna(row['age']):
            return row['age']
        
        # Try course-specific median
        course_median = course_age_medians.get(row['course_mapped'])
        if pd.notna(course_median):
            return course_median
        
        # Use global median
        return global_median_age
    
    return df.apply(fill_age, axis=1)

df['age_filled'] = impute_ages(df)

# Smart symptom handling
def clean_symptoms(symptom_val):
    if pd.isna(symptom_val) or str(symptom_val).strip() == "":
        return "general_consultation"  # Default symptom category
    return str(symptom_val).strip()

df['normalized_symptoms'] = df['normalized_symptoms'].apply(clean_symptoms)

# Create final cleaned dataset with no missing critical values
df_final = df.copy()
df_final['date_cleaned'] = pd.to_datetime(df_final['date_cleaned'])
df_final['age'] = df_final['age_filled']

# Quality check
print(f"\nData Quality Report:")
print(f"Total records: {len(df_final)}")
print(f"Records with dates: {df_final['date_cleaned'].notna().sum()}")
print(f"Records with courses: {df_final['course_mapped'].notna().sum()}")
print(f"Records with ages: {df_final['age'].notna().sum()}")
print(f"Records with symptoms: {(df_final['normalized_symptoms'] != 'general_consultation').sum()}")

print(f"\nMissing values after imputation:")
print(df_final[['date_cleaned', 'course_mapped', 'age', 'gender', 'normalized_symptoms']].isnull().sum())

# Select final columns
df_final = df_final[['date_cleaned', 'course_mapped', 'age', 'gender', 'normalized_symptoms']]

# Save cleaned file
name = input("File name: ")
df_final.to_csv(f'medisense/backend/data/cleaned/{name}.csv', index=False)

print(f"\n✅ Saved {len(df_final)} records to {name}.csv")