import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from rapidfuzz import process, fuzz

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 FUZZY LOGIC SYMPTOM MAPPING VISUALIZATION")
print("=" * 60)

# ==========================================
# 1. RECREATE YOUR MAPPING LOGIC
# ==========================================

# Load your data
df = pd.read_csv('medisense/backend/data/raw/merged.csv')

# Canonical symptoms list
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
    'abrasion': 'abrasion',
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
    # 'auri': 'cold',
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

# Recreate your cleaning and mapping process
def clean_symptom_string(text):
    import re
    text = text.lower()
    text = re.sub(r'[;/&]', ',', text)
    text = re.sub(r'\b(bp|hr|o2sat|rbs|t|temp|temperature|fbs)[^\s,]*', '', text)
    text = re.sub(r'\d+\.?\d*°?c?', '', text)
    text = re.sub(r'\d+/\d+', '', text)
    text = re.sub(r'[^\w\s,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r',+', ',', text)
    text = text.strip(' ,')
    return text

df['symptoms_cleaned'] = df['symptoms'].apply(clean_symptom_string)
df['symptom_list'] = df['symptoms_cleaned'].str.split(',')

# Get all unique symptoms
all_symptoms = set()
for sublist in df['symptom_list']:
    for item in sublist:
        token = item.strip()
        if token:
            all_symptoms.add(token)

# Create the mapping
symptom_map = {}
fuzzy_matches = []
manual_matches = []

for raw_symptom in all_symptoms:
    if raw_symptom in manual_map:
        symptom_map[raw_symptom] = manual_map[raw_symptom]
        manual_matches.append((raw_symptom, manual_map[raw_symptom], 100))
    else:
        match, score, _ = process.extractOne(
            raw_symptom, canonical_symptoms, scorer=fuzz.token_sort_ratio
        )
        if score > 80:
            symptom_map[raw_symptom] = match
            fuzzy_matches.append((raw_symptom, match, score))
        else:
            symptom_map[raw_symptom] = raw_symptom
            fuzzy_matches.append((raw_symptom, raw_symptom, score))

print(f"✅ Processed {len(all_symptoms)} unique symptoms")
print(f"   - Manual mappings: {len(manual_matches)}")
print(f"   - Fuzzy matches: {len(fuzzy_matches)}")

# ==========================================
# 2. MAPPING DISTRIBUTION ANALYSIS
# ==========================================

# Analyze mapping results
mapping_types = []
for raw, canonical in symptom_map.items():
    if raw in manual_map:
        mapping_types.append('Manual')
    elif raw == canonical:
        mapping_types.append('No Change')
    else:
        mapping_types.append('Fuzzy Match')

mapping_counts = Counter(mapping_types)

# ==========================================
# 3. CREATE VISUALIZATIONS
# ==========================================

# Create figure with subplots
fig = plt.figure(figsize=(20, 15))

# 1. Mapping Type Distribution
ax1 = plt.subplot(2, 3, 1)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
wedges, texts, autotexts = plt.pie(mapping_counts.values(), 
                                  labels=mapping_counts.keys(), 
                                  autopct='%1.1f%%',
                                  colors=colors,
                                  explode=(0.05, 0.05, 0.05))
plt.title('Symptom Mapping Distribution', fontsize=14, fontweight='bold')

# 2. Fuzzy Match Score Distribution
ax2 = plt.subplot(2, 3, 2)
fuzzy_scores = [score for _, _, score in fuzzy_matches if score < 100]
plt.hist(fuzzy_scores, bins=20, color='#45B7D1', alpha=0.7, edgecolor='black')
plt.xlabel('Fuzzy Match Score')
plt.ylabel('Frequency')
plt.title('Fuzzy Match Score Distribution', fontsize=14, fontweight='bold')
plt.axvline(x=80, color='red', linestyle='--', label='Threshold (80)')
plt.legend()

# 3. Top Canonical Symptoms (Target Distribution)
ax3 = plt.subplot(2, 3, 3)
canonical_counts = Counter(symptom_map.values())
top_10_canonical = canonical_counts.most_common(10)

symptoms, counts = zip(*top_10_canonical)
plt.barh(range(len(symptoms)), counts, color='#FF6B6B', alpha=0.7)
plt.yticks(range(len(symptoms)), symptoms)
plt.xlabel('Number of Raw Symptoms Mapped')
plt.title('Top 10 Target Canonical Symptoms', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 4. Mapping Quality Heatmap
ax4 = plt.subplot(2, 3, 4)
# Create categories for better visualization
score_ranges = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
range_labels = ['<60', '60-70', '70-80', '80-90', '90-100']
manual_scores = [100] * len(manual_matches)
all_scores = [score for _, _, score in fuzzy_matches] + manual_scores

score_dist = []
for low, high in score_ranges:
    count = sum(1 for score in all_scores if low <= score < high)
    score_dist.append(count)

# Add perfect matches (100)
perfect_count = sum(1 for score in all_scores if score == 100)
score_dist[-1] = perfect_count

# Create heatmap data
heatmap_data = np.array(score_dist).reshape(1, -1)
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=range_labels, yticklabels=['Count'], cbar_kws={'label': 'Frequency'})
plt.title('Mapping Quality Distribution', fontsize=14, fontweight='bold')

# 5. Network Graph of Sample Mappings
ax5 = plt.subplot(2, 3, 5)
# Create a sample network graph showing some mappings
G = nx.DiGraph()

# Select interesting mappings for visualization
sample_mappings = []
sample_mappings.extend(list(manual_map.items())[:8])  # First 8 manual mappings
sample_mappings.extend([(raw, canonical) for raw, canonical, score in fuzzy_matches[:8] if score > 80])

# Add nodes and edges
for raw, canonical in sample_mappings:
    G.add_edge(raw, canonical)

# Create layout
pos = nx.spring_layout(G, k=3, iterations=50)

# Draw network
raw_nodes = [node for node in G.nodes() if node in [raw for raw, _ in sample_mappings]]
canonical_nodes = [node for node in G.nodes() if node not in raw_nodes]

nx.draw_networkx_nodes(G, pos, nodelist=raw_nodes, node_color='lightblue', 
                      node_size=800, alpha=0.7)
nx.draw_networkx_nodes(G, pos, nodelist=canonical_nodes, node_color='lightcoral', 
                      node_size=1000, alpha=0.7)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

plt.title('Sample Symptom Mappings\n(Blue=Raw, Red=Canonical)', fontsize=14, fontweight='bold')
plt.axis('off')

# 6. Before/After Comparison
ax6 = plt.subplot(2, 3, 6)
# Count unique symptoms before and after normalization
unique_before = len(all_symptoms)
unique_after = len(set(symptom_map.values()))

categories = ['Before\nNormalization', 'After\nNormalization']
counts = [unique_before, unique_after]
reduction = ((unique_before - unique_after) / unique_before) * 100

bars = plt.bar(categories, counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
plt.ylabel('Number of Unique Symptoms')
plt.title(f'Vocabulary Reduction\n({reduction:.1f}% reduction)', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ==========================================
# 4. DETAILED MAPPING ANALYSIS
# ==========================================

print("\n📋 DETAILED MAPPING ANALYSIS")
print("-" * 40)

print(f"\n🎯 SUMMARY STATISTICS:")
print(f"   Total unique raw symptoms: {len(all_symptoms)}")
print(f"   Total canonical symptoms: {len(set(symptom_map.values()))}")
print(f"   Vocabulary reduction: {((len(all_symptoms) - len(set(symptom_map.values()))) / len(all_symptoms)) * 100:.1f}%")

print(f"\n📊 MAPPING BREAKDOWN:")
for mapping_type, count in mapping_counts.items():
    percentage = (count / len(all_symptoms)) * 100
    print(f"   {mapping_type}: {count} ({percentage:.1f}%)")

print(f"\n🔍 FUZZY MATCHING ANALYSIS:")
high_confidence = sum(1 for _, _, score in fuzzy_matches if score >= 90)
medium_confidence = sum(1 for _, _, score in fuzzy_matches if 80 <= score < 90)
low_confidence = sum(1 for _, _, score in fuzzy_matches if score < 80)

print(f"   High confidence (≥90): {high_confidence}")
print(f"   Medium confidence (80-89): {medium_confidence}")
print(f"   Low confidence (<80): {low_confidence}")

print(f"\n🏆 TOP CANONICAL TARGETS:")
for symptom, count in canonical_counts.most_common(5):
    print(f"   {symptom}: {count} raw symptoms mapped")

# Show some interesting mappings
print(f"\n🔗 SAMPLE MAPPINGS:")
print("   Manual mappings:")
for raw, canonical in list(manual_map.items())[:5]:
    print(f"      '{raw}' → '{canonical}'")

print("   High-confidence fuzzy matches:")
high_conf_fuzzy = [(raw, canonical, score) for raw, canonical, score in fuzzy_matches 
                   if score >= 90 and raw != canonical][:5]
for raw, canonical, score in high_conf_fuzzy:
    print(f"      '{raw}' → '{canonical}' (score: {score})")

print(f"\n✅ Visualization complete! Your fuzzy logic mapping has been analyzed.")