import pandas as pd
import re
from rapidfuzz import process, fuzz
from collections import defaultdict
import streamlit as st
import json
import os

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

# Your comprehensive manual mapping
MANUAL_MAP = {
    # Original corrections
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

    # Extended mapping for unmatched
    # Gastro / Abdominal
    'abdominal pain': 'stomach ache',
    'abdominal pain at defecating': 'stomach ache',
    'acid reflux': 'hyperacidity',
    'bloated': 'hyperacidity',
    'gassy': 'hyperacidity',
    'frequent burping': 'hyperacidity',
    'nausea': 'stomach ache',
    'upper stomach': 'stomach ache',
    'left upper quadrant abdominal pain': 'stomach ache',
    'left upper quadrant pain': 'stomach ache',
    'right upper quadrant abdominal pain': 'stomach ache',
    'upper quadrant pain': 'stomach ache',
    'hypogastric pain hyperacidity': 'hyperacidity',
    'hyperacidity epigastric pain': 'epigastric pain',
    'hyperacidity vomiting': 'vomiting',
    'lbm x days': 'lbm',
    'stomacheadacheche': 'stomach ache',

    # Respiratory
    'acute upper respiratory infection auri': 'auri',
    'acute upper respiratory infection aurti': 'auri',
    'auri': 'cold',
    'c acute upper respiratory infection auri': 'auri',
    'c acute upper respiratory infection aurti': 'auri',
    'c allergic cough': 'cough',
    'cold x days': 'cold',
    'colds x days': 'cold',
    'common cold x days': 'cold',
    'cough and colds x day': 'cough',
    'cough x days': 'cough',
    'persistent cough': 'cough',
    'productive cough': 'cough',
    'productive cough x weeks': 'cough',
    'dry cough': 'cough',
    'flu': 'cold',
    'flu vax': 'cold',
    'flu vax im': 'cold',
    'rhinitis': 'runny nose',
    'allergic rhinitis': 'runny nose',
    'sinusitis': 'runny nose',
    'weak lungs': 'asthma',
    'asthma attack': 'asthma',

    # Head-related
    'dizziness headache': 'headache',
    'headache dizziness': 'headache',
    'head ache': 'headache',
    'light headache': 'headache',
    'severe headache': 'headache',
    'headache at lack of sleep': 'headache',
    'slightly dizzy': 'dizziness',
    'vertigo': 'dizziness',
    'c vertigo': 'dizziness',

    # Reproductive
    'dysmenorrhea x days': 'dysmenorrhea',
    'pelvic pain': 'menstrual cramps',

    # Injury / Allergy
    'abraision left palm': 'abrasion',
    'abraison at fall': 'abrasion',
    'multiple abraision at fall': 'abrasion',
    'sprain': 'sprain',
    'ankle sprain': 'sprain',
    'knee sprain': 'sprain',
    'wrist sprain': 'sprain',
    'swoller ankle': 'sprain',
    'wrist swell': 'sprain',
    'knee swelling': 'sprain',
    'knee swoller': 'sprain',
    'shoulder dislocation': 'sprain',
    'shoulder swelling': 'sprain',
    'bruise': 'hematoma',
    'bruises': 'hematoma',
    'multiple bruises': 'hematoma',
    'hematoma': 'hematoma',
    'infected wound': 'infection',
    'inflamed punctured wound l foot': 'punctured wound',
    'index finger wound': 'wound',
    'small cut': 'cut',
    'lacerated wound': 'cut',
    'boils right foot': 'infection',
    'boils right lower quadrant': 'infection',
    'skin allergy at insect bite': 'skin allergy',
    'insect bite': 'skin allergy',
    'itchiness': 'skin allergy',
    'itchyness of': 'skin allergy',
    'nose bleed': 'nosebleed',
    'nose bleeding': 'nosebleed',
    'right knee abraises': 'abrasion',
    'rashes all over body': 'skin allergy',
    'chicken allergy': 'allergy',
    'burn': 'wound',
    'lw stiches': 'stitches',

    # General / Vitals
    'anxiety attack': 'anxiety',
    'body ache': 'body pain',
    'bodyache': 'body pain',
    'body weakness': 'malaise',
    'fatigue': 'malaise',
    'fainted': 'malaise',
    'malaise': 'malaise',
    'clammy skin': 'clammy skin',
    'highblood': 'hypertension',
    'hypertension': 'hypertension',
    'hypertensive': 'hypertension',
    'severe dehydration': 'malaise',
    'feeling warm bodypain': 'fever',
    'feeling warm headache': 'fever',
    'on and off fever': 'fever',
    'on and off fever x days': 'fever',
    'chills at night': 'fever',
    'c': 'fever',
    'o': 'fever',
    'sat': 'fever',
    'avr explode': 'malaise',
    'ccf': 'malaise',
    'post delivery': 'malaise',
    'for immunity': 'malaise',
    'underweight': 'malaise',
    'anemia': 'malaise',
    'lack of sleep': 'malaise',

    # Musculoskeletal
    'back pain': 'body pain',
    'backpain': 'body pain',
    'armpit': 'body pain',
    'lower back pain': 'body pain',
    'hip pain': 'body pain',
    'knee pain': 'body pain',
    'right leg pain': 'body pain',
    'bone pain': 'body pain',
    'muscle cramps': 'cramps',
    'minimal cramps': 'cramps',
    'cramps both upper': 'cramps',

    # Others
    'eye irritation': 'infection',
    'boils armpit': 'infection',
    'boil': 'infection',
    'gumpain': 'toothache',
    'extraction': 'toothache',
    'sore eyes': 'infection',
    'sore': 'infection',
    'lw m left inner': 'wound',
    'varicella': 'infection',
    'vomited x': 'vomiting',
    'dry bucalarea': 'dry mouth',
    'dysuria': 'uti',
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
    text = text.lower()
    text = text.replace("h/a", "headache")
    text = text.replace("ha", "headache")
    return text

def clean_symptom_string(text):
    """Your original cleaning function"""
    text = text.lower()
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
    """
    Complete symptom processing pipeline based on your script
    """
    canonical_symptoms, manual_map = load_dynamic_mappings()
    
    # Count raw unique symptoms
    unique_symptoms = df['symptoms'].unique()
    raw_count = len(unique_symptoms)
    
    # Apply your preprocessing
    df['symptoms_cleaned'] = df['symptoms'].apply(preprocess_known_abbreviations).apply(clean_symptom_string)
    
    # Count after cleaning
    unique_cleaned = df['symptoms_cleaned'].unique()
    cleaned_count = len(unique_cleaned)
    
    # Split into lists
    df['symptom_list'] = df['symptoms_cleaned'].str.split(',')
    
    # Flatten to get unique tokens
    all_symptoms = set()
    for sublist in df['symptom_list']:
        for item in sublist:
            token = item.strip()
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
    
    # Apply normalization
    df['normalized_symptoms'] = df['symptom_list'].apply(
        lambda symptoms: ",".join(sorted(set(
            symptom_map.get(sym.strip(), sym.strip()) for sym in symptoms if sym.strip()
        )))
    )
    
    # Clean up empty rows
    df = df.dropna(how='all')
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
        'unmatched_list': unmatched_symptoms
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