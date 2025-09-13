import pandas as pd
import re
from rapidfuzz import process, fuzz
from collections import defaultdict

df = pd.read_csv('medisense/data/raw/merged.csv')

#known abbreviation preprocessing

def preprocess_known_abbreviations(text):
    text = text.lower()
    text = text.replace("h/a", "headache")
    text = text.replace("ha", "headache")
    return text

# count raw unique symptoms
unique_symptoms = df['symptoms'].unique()
print(f"Total unique raw symptoms: {len(unique_symptoms)}")

#cleaning raw strings
def clean_symptom_string(text):
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

df['symptoms_cleaned'] = df['symptoms'].apply(preprocess_known_abbreviations).apply(clean_symptom_string)

# count symptoms after cleaning

unique = df['symptoms_cleaned'].unique()
print(f"Total unique cleaned symptom strings: {len(unique)}")

df['symptom_list'] = df['symptoms_cleaned'].str.split(',')

#flattening to get unique tokens
all_symptoms = set()
for sublist in df['symptom_list']:
    for item in sublist:
        token = item.strip()
        if token:
            all_symptoms.add(token)

#canonical symptoms list
canonical_symptoms = [
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

# manual_map = {
#     'dysminorrhea': 'dysmenorrhea',
#     'dizzy' : 'dizziness',
#     'allerfy': 'allergy',
#     'alergy': 'allergy',
#     'allergic': 'allergy',
#     'vomitting': 'vomiting',
#     'ons off fever': 'fever',
#     'off fever': 'fever',
#     'on': 'fever',
#     'cold clammyskin': 'clammy skin',
#     'midlle finger cut from bamboo': 'cut',
#     'lower extremeties': 'cramps',
#     'farm cramps': 'cramps',
#     'catching a cold several days': 'cold',
#     'feeling warm': 'fever',
#     'feeling warm 362c': 'fever',
#     'feeling warm 365c': 'fever',
#     'sick': 'malaise',
#     'itchy': 'itchy throat',
#     'h/a': 'headache',
#     'dob': 'shortness of breath',
#     'stiffneck': 'stiff neck',
#     'strain': 'muscle strain',
#     'abrasion': 'abrasion',  # typo fix
#     'stiches': 'stitches',
#     'pimples': 'pimple',
# }

manual_map = {
    # --- Original corrections ---
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
    'abrasion': 'abrasion',  # typo fix
    'stiches': 'stitches',
    'pimples': 'pimple',

    # --- Extended mapping for unmatched ---
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
    'flu vax': 'cold',   # optional, could move to misc
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


symptom_map = {}

#fuzzy matching for mapping symptoms to canonical list
for raw_symptom in all_symptoms:
    if raw_symptom in manual_map:
        symptom_map[raw_symptom] = manual_map[raw_symptom]
    else:
        match, score, _ = process.extractOne(
            raw_symptom, canonical_symptoms, scorer=fuzz.token_sort_ratio
        )
        if score > 80:
            symptom_map[raw_symptom] = match
        else:
            symptom_map[raw_symptom] = raw_symptom 

# df['normalized_symptoms'] = df['symptom_list'].apply(
#     lambda symptoms: sorted(set(
#         symptom_map.get(sym.strip(), sym.strip()) for sym in symptoms if sym.strip()
#     ))
# )

df['normalized_symptoms'] = df['symptom_list'].apply(
    lambda symptoms: ",".join(sorted(set(
        symptom_map.get(sym.strip(), sym.strip()) for sym in symptoms if sym.strip()
    )))
)

# debugging outputs

# print("\nNormalized Symptoms per Row:")
# count = 0
# for index, row in df.iterrows():
#     if row['normalized_symptoms'] == []:
#         count += 1
#     print(f"- Row {index + 2}: {row['normalized_symptoms']}")

# #empty string rows
# print(f"\nTotal rows with empty normalized symptoms: {count}")

# unmatched = {k: v for k, v in symptom_map.items() if v == k and k not in canonical_symptoms}

# if unmatched:
#     print("\nUnmatched or raw symptoms and their row numbers:")
# for raw in sorted(unmatched.keys()):
#     rows = df[df['symptoms'].str.strip().str.lower() == raw]  # filter df rows
#     row_numbers = rows.index.tolist()  # get row indices
#     print(f"- {raw} → rows {row_numbers}")



# # ensure canonical symptoms are unique
# canonical_symptoms = list(set(canonical_symptoms))

# # find bad targets (non existent in canonical)
# bad_targets = [v for v in manual_map.values() if v not in canonical_symptoms]

# if bad_targets:
#     print("\nBad targets in manual_map (not in canonical symptoms):")
#     for target in set(bad_targets):
#         print(f"- {target}")

# # reverse map to see if multiple raws map inconsistently

# reverse_map = defaultdict(list)
# for k, v in manual_map.items():
#     reverse_map[v].append(k)

# for target, raws in reverse_map.items():
#     if len(raws) > 1:
#         print(f"'{target}' ← {raws}")


df = df.dropna(how='all')

df = df[df['normalized_symptoms'].str.strip() != '']

df = df.reset_index(drop=True)

df = df[['date', 'course', 'gender', 'age', 'normalized_symptoms']]

name = input("File name: ")

df.to_csv(f'medisense/data/cleaned/{name}.csv', index=False)