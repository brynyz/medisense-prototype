import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import json
from collections import defaultdict
import re
from rapidfuzz import process, fuzz

st.set_page_config(
    page_title="MediSense Data Preprocessing",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your canonical symptoms list
CANONICAL_SYMPTOMS = [
    "headache", "dizziness", "fever", "cold", "cough", "sore throat",
    "asthma", "vomiting", "diarrhea", "cramps", "menstrual cramps",
    "toothache", "chest pain", "body pain", "hyperacidity", "anxiety", 
    "runny nose", "nosebleed", "stomach ache", "allergy", "migraine",
    "shortness of breath", "hyperventilation", "earache", "itchy throat",
    "abrasion", "wound", "skin allergy", "dysmenorrhea", "lbm", "epigastric pain",
    "punctured wound", "stiff neck", "infection", "cut", "pimple", "clammy skin",
    "dry mouth", "malaise", "uti", "hematoma", "muscle strain", "stitches", "auri",
    "sprain", "hypertension"
]

# Your manual mapping (from your script)
MANUAL_MAP = {
    'dysminorrhea': 'dysmenorrhea',
    'dizzy': 'dizziness',
    'allerfy': 'allergy',
    'alergy': 'allergy',
    'allergic': 'allergy',
    'vomitting': 'vomiting',
    'ons off fever': 'fever',
    'off fever': 'fever',
    'on': 'fever',
    'cold clammyskin': 'clammy skin',
    'midlle finger cut from bamboo': 'cut',
    'lower extremeties': 'cramps',
    'farm cramps': 'cramps',
    'catching a cold several days': 'cold',
    'feeling warm': 'fever',
    'feeling warm c': 'fever',
    'feeling warm 362c': 'fever',
    'feeling warm 365c': 'fever',
    'sick': 'malaise',
    'itchy': 'itchy throat',
    'h/a': 'headache',
    'dob': 'shortness of breath',
    'stiffneck': 'stiff neck',
    'strain': 'muscle strain',
    'abrasion': 'abrasion',
    'stiches': 'stitches',
    'pimples': 'pimple',
    # Add more from your script...
}

def load_dynamic_mappings():
    """Load dynamic canonical list and manual mappings from session state or files"""
    canonical_file = "canonical_symptoms.json"
    mapping_file = "manual_mappings.json"
    
    # Initialize from session state or defaults
    if 'canonical_symptoms' not in st.session_state:
        if os.path.exists(canonical_file):
            with open(canonical_file, 'r') as f:
                st.session_state['canonical_symptoms'] = json.load(f)
        else:
            st.session_state['canonical_symptoms'] = CANONICAL_SYMPTOMS.copy()
    
    if 'manual_mappings' not in st.session_state:
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                st.session_state['manual_mappings'] = json.load(f)
        else:
            st.session_state['manual_mappings'] = MANUAL_MAP.copy()
    
    return st.session_state['canonical_symptoms'], st.session_state['manual_mappings']

def save_dynamic_mappings():
    """Save canonical list and manual mappings to files"""
    canonical_file = "canonical_symptoms.json"
    mapping_file = "manual_mappings.json"
    
    with open(canonical_file, 'w') as f:
        json.dump(st.session_state['canonical_symptoms'], f, indent=2)
    
    with open(mapping_file, 'w') as f:
        json.dump(st.session_state['manual_mappings'], f, indent=2)

def preprocess_known_abbreviations(text):
    """Your original preprocessing function"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = text.replace("h/a", "headache")
    text = text.replace("ha", "headache")
    return text

def clean_symptom_string(text):
    """Your original cleaning function"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[;/&]', ',', text)  # unify delimiters
    text = re.sub(r'\b(bp|hr|o2sat|rbs|t|temp|temperature|fbs)[^\s,]*', '', text)  # vitals
    text = re.sub(r'\d+\.?\d*°?c?', '', text)  # temperatures
    text = re.sub(r'\d+/\d+', '', text)  # BP
    text = re.sub(r'[^\w\s,]', '', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text)  # clean spaces
    text = re.sub(r'\b\d+\b', '', text)  # standalone numbers
    text = re.sub(r',+', ',', text)  # multiple commas
    text = text.strip(' ,')
    return text

def process_symptoms_full_pipeline(df, fuzzy_threshold=80):
    """Complete symptom processing pipeline based on your script"""
    canonical_symptoms, manual_map = load_dynamic_mappings()
    
    # Check for symptoms column and handle different possible names
    symptom_column = None
    possible_symptom_columns = ['symptoms', 'symptom', 'chief_complaint', 'complaint', 'presenting_complaint', 'chief complaint', 'remarks']
    
    for col in possible_symptom_columns:
        if col in df.columns:
            symptom_column = col
            break
    
    if symptom_column is None:
        # If no standard symptom column found, show available columns
        available_cols = list(df.columns)
        raise ValueError(f"No symptoms column found. Available columns: {available_cols}")
    
    # Count raw unique symptoms
    unique_symptoms = df[symptom_column].dropna().unique()
    raw_count = len(unique_symptoms)
    
    # Apply your preprocessing
    df['symptoms_cleaned'] = df[symptom_column].apply(preprocess_known_abbreviations).apply(clean_symptom_string)
    
    # Count after cleaning
    unique_cleaned = df['symptoms_cleaned'].dropna().unique()
    cleaned_count = len(unique_cleaned)
    
    # Split into lists
    df['symptom_list'] = df['symptoms_cleaned'].str.split(',')
    
    # Flatten to get unique tokens
    all_symptoms = set()
    for sublist in df['symptom_list'].dropna():
        if isinstance(sublist, list):  # Make sure it's a list
            for item in sublist:
                token = str(item).strip() if item else ""
                if token:
                    all_symptoms.add(token)
    
    # Create symptom map using your logic
    symptom_map = {}
    unmatched_symptoms = []
    
    for raw_symptom in all_symptoms:
        if raw_symptom in manual_map:
            symptom_map[raw_symptom] = manual_map[raw_symptom]
        else:
            match, score, _ = process.extractOne(
                raw_symptom, canonical_symptoms, scorer=fuzz.token_sort_ratio
            )
            if score > fuzzy_threshold:
                symptom_map[raw_symptom] = match
            else:
                symptom_map[raw_symptom] = raw_symptom
                if raw_symptom not in canonical_symptoms:
                    unmatched_symptoms.append(raw_symptom)
    
    # Apply normalization - FIXED FUNCTION
    def normalize_symptom_list(symptoms):
        # Handle different types of input
        if symptoms is None:
            return ""
        
        # If it's not a list, try to make it one
        if not isinstance(symptoms, list):
            if pd.isna(symptoms):
                return ""
            # Convert string to list if needed
            if isinstance(symptoms, str):
                symptoms = [symptoms] if symptoms.strip() else []
            else:
                return ""
        
        # If it's an empty list
        if len(symptoms) == 0:
            return ""
        
        normalized = []
        for sym in symptoms:
            if sym is not None and not pd.isna(sym):
                sym_clean = str(sym).strip()
                if sym_clean:  # Only process non-empty strings
                    mapped = symptom_map.get(sym_clean, sym_clean)
                    normalized.append(mapped)
        
        return ",".join(sorted(set(normalized))) if normalized else ""
    
    df['normalized_symptoms'] = df['symptom_list'].apply(normalize_symptom_list)
    
    # Clean up empty rows
    df = df.dropna(subset=['normalized_symptoms'])
    df = df[df['normalized_symptoms'].str.strip() != '']
    df = df.reset_index(drop=True)
    
    # Prepare result statistics
    processing_stats = {
        'raw_unique_symptoms': raw_count,
        'cleaned_unique_symptoms': cleaned_count,
        'total_unique_tokens': len(all_symptoms),
        'mapped_symptoms': len([s for s in symptom_map.values() if s in canonical_symptoms]),
        'unmatched_symptoms': len(unmatched_symptoms),
        'final_rows': len(df),
        'symptom_map': symptom_map,
        'unmatched_list': unmatched_symptoms,
        'symptom_column_used': symptom_column  # Track which column was used
    }
    
    return df, processing_stats

def get_mapping_suggestions(symptom, canonical_symptoms, num_suggestions=5):
    """Get fuzzy matching suggestions for a symptom"""
    suggestions = process.extract(
        symptom, 
        canonical_symptoms, 
        scorer=fuzz.token_sort_ratio, 
        limit=num_suggestions
    )
    return [(match, score) for match, score, _ in suggestions]

def show_data_upload():
    """Data Upload and Initial Analysis"""
    st.markdown("### 📁 Data Upload & Initial Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your medical records CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: date, course, gender, age, symptoms, remarks"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state['raw_data'] = df
            
            st.success(f"✅ Successfully loaded {len(df):,} records")
            
            # Quick data overview with proper type conversion
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                missing_count = int(df.isnull().sum().sum())
                st.metric("Missing Values", f"{missing_count:,}")
            with col4:
                duplicate_count = int(df.duplicated().sum())
                st.metric("Duplicates", f"{duplicate_count:,}")
            
            # Column analysis
            st.markdown("#### 📋 Column Analysis")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique(),
                'Sample Value': [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Data preview
            with st.expander("👀 Raw Data Preview (First 20 rows)", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
            
            # Data quality issues detection
            st.markdown("#### 🚨 Data Quality Issues Detected")
            
            issues = detect_data_issues(df)
            
            if issues:
                for issue in issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ No major data quality issues detected!")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted")
    
    else:
        # Show expected format
        st.markdown("#### 📝 Expected Data Format")
        st.info("Please upload a CSV file with the following structure:")
        
        sample_data = pd.DataFrame({
            'date': ['2025-01-15', '2025-01-16', '2025-01-17'],
            'course': ['BSCS', 'BSN', 'BSE'],
            'gender': ['Male', 'Female', 'Male'],
            'age': [20, 19, 21],
            'symptoms': ['Headache, Fatigue', 'Nausea, Dizziness', 'Fever, Cough'],
            'remarks': ['Stress-related', 'Food poisoning suspected', 'Viral infection']
        })
        
        st.dataframe(sample_data, use_container_width=True)

def detect_data_issues(df):
    """Detect common data quality issues"""
    issues = []
    
    # Check for required columns (more flexible)
    required_cols = ['date', 'course', 'gender', 'age']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for symptom-related columns
    symptom_cols = [col for col in df.columns if 'symptom' in col.lower() or 'complaint' in col.lower() or 'remark' in col.lower()]
    if not symptom_cols:
        issues.append("No symptom-related columns found (should have 'symptoms', 'symptom', 'complaint', or 'remarks' column)")
    
    # Check date format issues
    if 'date' in df.columns:
        date_issues = int(df['date'].astype(str).str.contains(r'[^0-9/-]', regex=True).sum())
        if date_issues > 0:
            issues.append(f"{date_issues} rows have invalid date formats")
    
    # Check age issues
    if 'age' in df.columns:
        invalid_ages = int(df['age'].apply(lambda x: pd.isna(x) or (isinstance(x, (int, float)) and (x < 16 or x > 100))).sum())
        if invalid_ages > 0:
            issues.append(f"{invalid_ages} rows have invalid ages (should be 16-100)")
    
    # Check empty symptoms in any symptom column
    for symptom_col in symptom_cols:
        empty_symptoms = int(df[symptom_col].isna().sum() + (df[symptom_col] == '').sum())
        if empty_symptoms > 0:
            issues.append(f"{empty_symptoms} rows have empty values in '{symptom_col}' column")
    
    return issues

def show_data_cleaning():
    """Data Cleaning Interface"""
    st.markdown("### 🧹 Data Cleaning & Standardization")
    
    if 'raw_data' not in st.session_state:
        st.warning("⚠️ Please upload data first in the Data Upload tab")
        return
    
    df = st.session_state['raw_data']
    
    # Cleaning options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Basic Cleaning")
        remove_duplicates = st.checkbox("Remove duplicate records", value=True)
        handle_missing = st.selectbox(
            "Handle missing values",
            ["Keep as is", "Drop rows with missing required fields", "Fill with defaults"]
        )
        standardize_text = st.checkbox("Standardize text case and spacing", value=True)
    
    with col2:
        st.markdown("#### ✅ Validation Rules")
        validate_dates = st.checkbox("Validate and fix date formats", value=True)
        validate_ages = st.checkbox("Validate age ranges (16-100)", value=True)
        map_courses = st.checkbox("Map course abbreviations to full names", value=True)
        
    # Apply cleaning
    if st.button("🚀 Apply Data Cleaning", type="primary"):
        with st.spinner("Cleaning data..."):
            cleaned_df, cleaning_report = validate_and_clean_data(
                df, remove_duplicates, handle_missing, standardize_text,
                validate_dates, validate_ages, map_courses
            )
            
            # Store cleaned data
            st.session_state['cleaned_data'] = cleaned_df
            st.session_state['cleaning_report'] = cleaning_report
            
            # Show results
            st.success("✅ Data cleaning completed!")
            
            # Before/After comparison with proper type conversion
            col1, col2, col3 = st.columns(3)
            
            with col1:
                records_delta = int(cleaning_report['final_rows']) - int(cleaning_report['original_rows'])
                st.metric(
                    "Records", 
                    f"{cleaning_report['final_rows']:,}",
                    delta=records_delta
                )
            
            with col2:
                original_missing = int(df.isnull().sum().sum())
                final_missing = int(cleaned_df.isnull().sum().sum())
                missing_delta = final_missing - original_missing
                st.metric(
                    "Missing Values",
                    f"{final_missing:,}",
                    delta=missing_delta
                )
            
            with col3:
                duplicates_removed = int(cleaning_report['original_rows']) - int(cleaning_report['final_rows'])
                st.metric("Duplicates Removed", f"{duplicates_removed:,}")
            
            # Show cleaning report
            with st.expander("📋 Detailed Cleaning Report", expanded=True):
                st.markdown("**Issues Found:**")
                for issue in cleaning_report['issues_found']:
                    st.write(f"• {issue}")
                
                st.markdown("**Fixes Applied:**")
                for fix in cleaning_report['fixes_applied']:
                    st.write(f"✅ {fix}")
            
            # Show cleaned data preview
            with st.expander("👀 Cleaned Data Preview", expanded=False):
                st.dataframe(cleaned_df.head(20), use_container_width=True)
    
    # Show current cleaning status
    if 'cleaned_data' in st.session_state:
        st.info("✅ Data has been cleaned and is ready for symptom mapping")

def validate_and_clean_data(df, remove_duplicates, handle_missing, standardize_text,
                           validate_dates, validate_ages, map_courses):
    """Main data validation and cleaning function"""
    cleaned_df = df.copy()
    cleaning_report = {
        'original_rows': len(df),
        'issues_found': [],
        'fixes_applied': []
    }
    
    # Remove duplicates
    if remove_duplicates:
        duplicates = cleaned_df.duplicated().sum()
        if duplicates > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            cleaning_report['fixes_applied'].append(f"Removed {duplicates} duplicate rows")
    
    # Handle missing values
    if handle_missing == "Drop rows with missing required fields":
        required_cols = ['date', 'course', 'gender', 'age', 'symptoms']
        before_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=[col for col in required_cols if col in cleaned_df.columns])
        dropped_rows = before_rows - len(cleaned_df)
        if dropped_rows > 0:
            cleaning_report['fixes_applied'].append(f"Dropped {dropped_rows} rows with missing required fields")
    
    elif handle_missing == "Fill with defaults":
        if 'remarks' in cleaned_df.columns:
            cleaned_df['remarks'] = cleaned_df['remarks'].fillna('')
        if 'age' in cleaned_df.columns:
            cleaned_df['age'] = pd.to_numeric(cleaned_df['age'], errors='coerce')
            median_age = cleaned_df['age'].median()
            filled_ages = cleaned_df['age'].isna().sum()
            cleaned_df['age'] = cleaned_df['age'].fillna(median_age)
            if filled_ages > 0:
                cleaning_report['fixes_applied'].append(f"Filled {filled_ages} missing ages with median ({median_age})")
    
    # Standardize text
    if standardize_text:
        text_columns = ['course', 'symptoms', 'remarks']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                cleaning_report['fixes_applied'].append(f"Standardized text in {col} column")
    
    # Validate dates
    if validate_dates and 'date' in cleaned_df.columns:
        before_count = len(cleaned_df)
        cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
        invalid_dates = cleaned_df['date'].isna().sum()
        if invalid_dates > 0:
            cleaning_report['issues_found'].append(f"Found {invalid_dates} invalid dates")
            cleaned_df = cleaned_df[cleaned_df['date'].notna()]
        
        # Convert back to string format
        cleaned_df['date'] = cleaned_df['date'].dt.strftime('%Y-%m-%d')
        date_rows_removed = before_count - len(cleaned_df)
        if date_rows_removed > 0:
            cleaning_report['fixes_applied'].append(f"Removed {date_rows_removed} rows with invalid dates")
    
    # Validate ages
    if validate_ages and 'age' in cleaned_df.columns:
        cleaned_df['age'] = pd.to_numeric(cleaned_df['age'], errors='coerce')
        before_count = len(cleaned_df)
        cleaned_df = cleaned_df[(cleaned_df['age'] >= 16) & (cleaned_df['age'] <= 100)]
        age_rows_removed = before_count - len(cleaned_df)
        if age_rows_removed > 0:
            cleaning_report['fixes_applied'].append(f"Removed {age_rows_removed} rows with invalid ages")
    
    # Map courses
    if map_courses and 'course' in cleaned_df.columns:
        course_mapping = {
            'BSCS': 'Computer Science',
            'BSIT': 'Information Technology', 
            'BSBA': 'Business Administration',
            'BEED': 'Elementary Education',
            'BSE': 'Engineering',
            'BSCRIM': 'Criminology',
            'BSLM': 'Laboratory Medicine',
            'BSTM': 'Tourism Management',
            'BSENT': 'Entrepreneurship',
            'BSHM': 'Hotel Management'
        }
        
        mapped_count = 0
        for abbrev, full_name in course_mapping.items():
            mask = cleaned_df['course'].str.upper() == abbrev
            if mask.any():
                cleaned_df.loc[mask, 'course'] = full_name
                mapped_count += mask.sum()
        
        if mapped_count > 0:
            cleaning_report['fixes_applied'].append(f"Mapped {mapped_count} course abbreviations to full names")
    
    cleaning_report['final_rows'] = len(cleaned_df)
    
    return cleaned_df, cleaning_report

def calculate_quality_score(df):
    """Calculate a data quality score out of 10"""
    score = 0.0
    
    # Completeness (40% of score)
    completeness = float(df.count().sum() / df.size) * 4
    score += completeness
    
    # Age validity (20% of score)
    if 'age' in df.columns:
        valid_ages = float(df['age'].between(16, 100).sum() / len(df)) * 2
        score += valid_ages
    
    # Date validity (20% of score)
    if 'date' in df.columns:
        valid_dates = float(pd.to_datetime(df['date'], errors='coerce').notna().sum() / len(df)) * 2
        score += valid_dates
    
    # Symptoms mapped (20% of score)
    if 'normalized_symptoms' in df.columns:
        mapped_symptoms = float(df['normalized_symptoms'].notna().sum() / len(df)) * 2
        score += mapped_symptoms
    
    return min(float(score), 10.0)

def main():
    st.header("🔧 MediSense Data Preprocessing")
    st.subheader("Clean and standardize medical data using advanced NLP and fuzzy logic")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data Upload", 
        "🧹 Data Cleaning", 
        "🔍 Symptom Mapping", 
        "📊 Quality Report"
    ])
    
    with tab1:
        show_data_upload()
    
    with tab2:
        show_data_cleaning()
    
    with tab3:
        show_symptom_mapping()
    
    with tab4:
        show_quality_report()

def show_symptom_mapping():
    """Enhanced Symptom Mapping with Dynamic Management"""
    st.markdown("### 🔍 Intelligent Symptom Mapping & Management")
    
    if 'cleaned_data' not in st.session_state:
        st.warning("⚠️ Please clean your data first")
        return
    
    df = st.session_state['cleaned_data'].copy()
    
    # Check for symptom columns
    symptom_columns = [col for col in df.columns if 'symptom' in col.lower() or 'complaint' in col.lower() or 'remark' in col.lower()]
    
    if not symptom_columns:
        st.error("❌ No symptom-related columns found in your data!")
        st.write("**Available columns:**")
        for col in df.columns:
            st.write(f"• {col}")
        st.info("💡 Please ensure your CSV has a column containing symptom information")
        return
    
    # Show detected symptom columns
    if len(symptom_columns) > 1:
        st.info(f"🔍 Found multiple symptom columns: {', '.join(symptom_columns)}")
        st.info("The system will automatically use the most appropriate column")
    else:
        st.info(f"✅ Found symptom column: **{symptom_columns[0]}**")
    
    # Load dynamic mappings
    canonical_symptoms, manual_mappings = load_dynamic_mappings()
    
    # Create tabs for different functionality
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Symptom Processing", 
        "📋 Canonical List Management", 
        "🔗 Manual Mapping Interface",
        "📊 Mapping Statistics"
    ])
    
    with tab1:
        st.markdown("#### 🧠 Automated Symptom Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fuzzy_threshold = st.slider(
                "Fuzzy Matching Threshold", 
                60, 100, 80, 
                help="Higher values = stricter matching"
            )
            
            st.info(f"Using {len(canonical_symptoms)} canonical symptoms")
            st.info(f"Using {len(manual_mappings)} manual mappings")
        
        with col2:
            st.markdown("**Processing Pipeline:**")
            st.write("1. 🔤 Preprocess abbreviations (h/a → headache)")
            st.write("2. 🧹 Clean text (remove vitals, special chars)")
            st.write("3. 🎯 Apply manual mappings")
            st.write("4. 🔍 Fuzzy match to canonical symptoms")
            st.write("5. ✅ Normalize and deduplicate")
        
        # Process symptoms button
        if st.button("🚀 Process All Symptoms", type="primary"):
            with st.spinner("Processing symptoms with full pipeline..."):
                processed_df, stats = process_symptoms_full_pipeline(df, fuzzy_threshold)
                
                # Store results
                st.session_state['processed_symptoms_df'] = processed_df
                st.session_state['processing_stats'] = stats
                
                # Show results
                st.success("✅ Symptom processing completed!")
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Raw Unique Symptoms", f"{stats['raw_unique_symptoms']:,}")
                
                with col2:
                    st.metric("After Cleaning", f"{stats['cleaned_unique_symptoms']:,}")
                
                with col3:
                    st.metric("Total Tokens", f"{stats['total_unique_tokens']:,}")
                
                with col4:
                    st.metric("Successfully Mapped", f"{stats['mapped_symptoms']:,}")
                
                # Show unmatched symptoms
                if stats['unmatched_list']:
                    st.warning(f"⚠️ {len(stats['unmatched_list'])} symptoms remain unmatched")
                    
                    with st.expander("👀 View Unmatched Symptoms", expanded=False):
                        unmatched_df = pd.DataFrame({
                            'Unmatched Symptom': stats['unmatched_list'],
                            'Frequency': [
                                sum(1 for _, row in processed_df.iterrows() 
                                    if symptom in row['normalized_symptoms']) 
                                for symptom in stats['unmatched_list']
                            ]
                        }).sort_values('Frequency', ascending=False)
                        
                        st.dataframe(unmatched_df, use_container_width=True)
                
                # Show sample processed data
                with st.expander("📋 Processed Data Preview", expanded=True):
                    display_df = processed_df[['symptoms', 'symptoms_cleaned', 'normalized_symptoms']].head(10)
                    st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        st.markdown("#### 📋 Canonical Symptoms Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Current Canonical Symptoms:**")
            
            # Display current canonical symptoms in a searchable format
            search_canonical = st.text_input("🔍 Search canonical symptoms:", placeholder="Type to filter...")
            
            if search_canonical:
                filtered_canonical = [s for s in canonical_symptoms if search_canonical.lower() in s.lower()]
            else:
                filtered_canonical = canonical_symptoms
            
            # Display in columns for better readability
            chunks = [filtered_canonical[i:i+3] for i in range(0, len(filtered_canonical), 3)]
            for chunk in chunks:
                cols = st.columns(3)
                for i, symptom in enumerate(chunk):
                    with cols[i]:
                        # Add delete button for each symptom
                        if st.button(f"❌", key=f"del_canonical_{symptom}"):
                            canonical_symptoms.remove(symptom)
                            st.session_state['canonical_symptoms'] = canonical_symptoms
                            save_dynamic_mappings()
                            st.rerun()
                        st.write(f"• {symptom}")
        
        with col2:
            st.markdown("**Add New Canonical Symptom:**")
            
            new_canonical = st.text_input(
                "New canonical symptom:",
                placeholder="Enter new symptom..."
            )
            
            if st.button("➕ Add to Canonical List"):
                if new_canonical and new_canonical.lower() not in [s.lower() for s in canonical_symptoms]:
                    canonical_symptoms.append(new_canonical.lower().strip())
                    st.session_state['canonical_symptoms'] = sorted(canonical_symptoms)
                    save_dynamic_mappings()
                    st.success(f"✅ Added '{new_canonical}' to canonical list")
                    st.rerun()
                elif new_canonical.lower() in [s.lower() for s in canonical_symptoms]:
                    st.warning("⚠️ Symptom already exists in canonical list")
                else:
                    st.warning("⚠️ Please enter a valid symptom")
            
            # Bulk import
            st.markdown("**Bulk Import:**")
            bulk_symptoms = st.text_area(
                "Add multiple symptoms (one per line):",
                placeholder="headache\nfever\ncough\n..."
            )
            
            if st.button("📥 Bulk Import"):
                if bulk_symptoms:
                    new_symptoms = [s.strip().lower() for s in bulk_symptoms.split('\n') if s.strip()]
                    added_count = 0
                    
                    for symptom in new_symptoms:
                        if symptom not in [s.lower() for s in canonical_symptoms]:
                            canonical_symptoms.append(symptom)
                            added_count += 1
                    
                    st.session_state['canonical_symptoms'] = sorted(canonical_symptoms)
                    save_dynamic_mappings()
                    st.success(f"✅ Added {added_count} new symptoms to canonical list")
                    st.rerun()
    
    with tab3:
        st.markdown("#### 🔗 Manual Mapping Interface")
        
        # Show current manual mappings
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Current Manual Mappings:**")
            
            if manual_mappings:
                # Convert to DataFrame for display
                mapping_df = pd.DataFrame([
                    {'Raw Symptom': k, 'Maps To': v} 
                    for k, v in manual_mappings.items()
                ])
                
                # Add search functionality
                search_mapping = st.text_input("🔍 Search mappings:", placeholder="Type to filter...")
                
                if search_mapping:
                    mask = (mapping_df['Raw Symptom'].str.contains(search_mapping, case=False) | 
                           mapping_df['Maps To'].str.contains(search_mapping, case=False))
                    filtered_mapping_df = mapping_df[mask]
                else:
                    filtered_mapping_df = mapping_df
                
                # Display with delete options
                for idx, row in filtered_mapping_df.iterrows():
                    col_a, col_b, col_c = st.columns([3, 3, 1])
                    
                    with col_a:
                        st.write(f"**{row['Raw Symptom']}**")
                    
                    with col_b:
                        st.write(f"→ {row['Maps To']}")
                    
                    with col_c:
                        if st.button("🗑️", key=f"del_mapping_{idx}"):
                            del manual_mappings[row['Raw Symptom']]
                            st.session_state['manual_mappings'] = manual_mappings
                            save_dynamic_mappings()
                            st.rerun()
            else:
                st.info("No manual mappings yet.")
        
        with col2:
            st.markdown("**Add New Manual Mapping:**")
            
            # If we have unmatched symptoms, show them as options
            if 'processing_stats' in st.session_state and st.session_state['processing_stats']['unmatched_list']:
                unmatched_options = ["Type custom..."] + st.session_state['processing_stats']['unmatched_list']
                
                selected_unmatched = st.selectbox(
                    "Select unmatched symptom:",
                    unmatched_options
                )
                
                if selected_unmatched == "Type custom...":
                    raw_symptom = st.text_input("Raw symptom:", placeholder="Enter raw symptom...")
                else:
                    raw_symptom = selected_unmatched
                    st.write(f"Selected: **{raw_symptom}**")
                    
                    # Show suggestions for this symptom
                    if raw_symptom:
                        suggestions = get_mapping_suggestions(raw_symptom, canonical_symptoms)
                        st.write("**Suggested mappings:**")
                        for match, score in suggestions[:3]:
                            st.write(f"• {match} ({score}% match)")
            else:
                raw_symptom = st.text_input("Raw symptom:", placeholder="Enter raw symptom...")
            
            # Target canonical symptom
            target_symptom = st.selectbox(
                "Map to canonical symptom:",
                [""] + canonical_symptoms
            )
            
            if st.button("🔗 Add Mapping"):
                if raw_symptom and target_symptom:
                    manual_mappings[raw_symptom.lower().strip()] = target_symptom
                    st.session_state['manual_mappings'] = manual_mappings
                    save_dynamic_mappings()
                    st.success(f"✅ Mapped '{raw_symptom}' → '{target_symptom}'")
                    st.rerun()
                else:
                    st.warning("⚠️ Please fill in both fields")
            
            # Quick add from unmatched
            if ('processing_stats' in st.session_state and 
                st.session_state['processing_stats']['unmatched_list']):
                
                st.markdown("**Quick Mapping Assistant:**")
                
                for unmatched in st.session_state['processing_stats']['unmatched_list'][:5]:
                    with st.expander(f"Map: {unmatched}"):
                        suggestions = get_mapping_suggestions(unmatched, canonical_symptoms, 3)
                        
                        for i, (match, score) in enumerate(suggestions):
                            if st.button(f"Map to '{match}' ({score}%)", key=f"quick_map_{unmatched}_{i}"):
                                manual_mappings[unmatched] = match
                                st.session_state['manual_mappings'] = manual_mappings
                                save_dynamic_mappings()
                                st.success(f"✅ Quick mapped '{unmatched}' → '{match}'")
                                st.rerun()
    
    with tab4:
        st.markdown("#### 📊 Processing Statistics & Validation")
        
        if 'processing_stats' in st.session_state:
            stats = st.session_state['processing_stats']
            
            # Overall statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Processing Efficiency", f"{(stats['mapped_symptoms']/stats['total_unique_tokens']*100):.1f}%")
            
            with col2:
                st.metric("Canonical Coverage", f"{len(canonical_symptoms)} symptoms")
            
            with col3:
                st.metric("Manual Mappings", f"{len(manual_mappings)} rules")
            
            # Validation checks
            st.markdown("#### ✅ Validation Report")
            
            # Check for bad targets in manual mappings
            bad_targets = [v for v in manual_mappings.values() if v not in canonical_symptoms]
            
            if bad_targets:
                st.error(f"❌ Found {len(bad_targets)} manual mappings pointing to non-canonical symptoms:")
                for target in set(bad_targets):
                    st.write(f"• '{target}' is not in canonical list")
            else:
                st.success("✅ All manual mappings point to valid canonical symptoms")
            
            # Check for mapping conflicts
            reverse_map = defaultdict(list)
            for k, v in manual_mappings.items():
                reverse_map[v].append(k)
            
            conflicts = {target: raws for target, raws in reverse_map.items() if len(raws) > 1}
            
            if conflicts:
                st.warning(f"⚠️ Found {len(conflicts)} canonical symptoms with multiple mappings:")
                for target, raws in conflicts.items():
                    st.write(f"• '{target}' ← {raws}")
            else:
                st.success("✅ No mapping conflicts detected")
            
            # Most common unmatched symptoms
            if stats['unmatched_list']:
                st.markdown("#### 🎯 Top Unmatched Symptoms to Address")
                
                if 'processed_symptoms_df' in st.session_state:
                    df_processed = st.session_state['processed_symptoms_df']
                    
                    unmatched_freq = {}
                    for symptom in stats['unmatched_list']:
                        freq = sum(1 for _, row in df_processed.iterrows() 
                                  if symptom in row['normalized_symptoms'])
                        unmatched_freq[symptom] = freq
                    
                    top_unmatched = sorted(unmatched_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    unmatched_chart_df = pd.DataFrame(top_unmatched, columns=['Symptom', 'Frequency'])
                    
                    fig = px.bar(
                        unmatched_chart_df,
                        x='Frequency',
                        y='Symptom',
                        orientation='h',
                        title="Top 10 Unmatched Symptoms by Frequency"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("🔄 Run symptom processing first to see statistics")
        
        # Download configuration
        st.markdown("#### 💾 Export Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            canonical_json = json.dumps(canonical_symptoms, indent=2)
            st.download_button(
                "📥 Download Canonical List",
                data=canonical_json,
                file_name="canonical_symptoms.json",
                mime="application/json"
            )
        
        with col2:
            mappings_json = json.dumps(manual_mappings, indent=2)
            st.download_button(
                "📥 Download Manual Mappings",
                data=mappings_json,
                file_name="manual_mappings.json",
                mime="application/json"
            )

def show_quality_report():
    """Enhanced Quality Report with Symptom Processing"""
    st.markdown("### 📊 Final Data Quality Report")
    
    if 'processed_symptoms_df' not in st.session_state:
        st.warning("⚠️ Please complete the full symptom processing pipeline first")
        return
    
    df = st.session_state['processed_symptoms_df']
    stats = st.session_state.get('processing_stats', {})
    
    # Final statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Records", f"{len(df):,}")
    
    with col2:
        st.metric("Symptom Processing", f"{stats.get('mapped_symptoms', 0)}/{stats.get('total_unique_tokens', 0)}")
    
    with col3:
        if stats:
            efficiency = (stats.get('mapped_symptoms', 0) / max(stats.get('total_unique_tokens', 1), 1)) * 100
            st.metric("Processing Efficiency", f"{efficiency:.1f}%")
        else:
            st.metric("Processing Efficiency", "N/A")
    
    with col4:
        quality_score = calculate_quality_score(df)
        st.metric("Overall Quality Score", f"{quality_score:.1f}/10")
    
    # Show final processed data structure
    st.markdown("#### 📋 Final Processed Data Preview")
    
    display_columns = ['date', 'course', 'gender', 'age', 'symptoms', 'normalized_symptoms']
    available_columns = [col for col in display_columns if col in df.columns]
    
    st.dataframe(df[available_columns].head(20), use_container_width=True)
    
    # Processing pipeline summary
    with st.expander("📈 Complete Processing Pipeline Summary", expanded=True):
        if 'cleaning_report' in st.session_state:
            cleaning_report = st.session_state['cleaning_report']
            
            st.markdown("**Data Cleaning Results:**")
            st.write(f"• Original records: {cleaning_report.get('original_rows', 'N/A'):,}")
            st.write(f"• After cleaning: {cleaning_report.get('final_rows', 'N/A'):,}")
            
            if 'fixes_applied' in cleaning_report:
                st.write("• Fixes applied:")
                for fix in cleaning_report['fixes_applied']:
                    st.write(f"  - {fix}")
        
        if stats:
            st.markdown("**Symptom Processing Results:**")
            st.write(f"• Raw unique symptoms: {stats.get('raw_unique_symptoms', 'N/A'):,}")
            st.write(f"• After text cleaning: {stats.get('cleaned_unique_symptoms', 'N/A'):,}")
            st.write(f"• Unique symptom tokens: {stats.get('total_unique_tokens', 'N/A'):,}")
            st.write(f"• Successfully mapped: {stats.get('mapped_symptoms', 'N/A'):,}")
            st.write(f"• Remaining unmatched: {stats.get('unmatched_symptoms', 'N/A'):,}")
    
    # Export final processed data
    st.markdown("### 💾 Export Final Processed Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"medisense_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export
        json_data = df.to_json(orient='records', date_format='iso')
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"medisense_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Processing report export
        if stats:
            report = {
                'processing_date': datetime.now().isoformat(),
                'statistics': stats,
                'cleaning_report': st.session_state.get('cleaning_report', {}),
                'canonical_symptoms_count': len(st.session_state.get('canonical_symptoms', [])),
                'manual_mappings_count': len(st.session_state.get('manual_mappings', {}))
            }
            
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="📥 Download Processing Report",
                data=report_json,
                file_name=f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()