"""
Diagnostic Script for Perfect Accuracy Issue
=============================================
This script identifies why the model is achieving perfect accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("🔍 DIAGNOSING PERFECT ACCURACY ISSUE")
print("=" * 70)

# Load the dataset
dataset_path = 'medisense/backend/data/final/optimized/dataset_binary_visits_optimized.csv'
try:
    df = pd.read_csv(dataset_path)
except:
    dataset_path = 'medisense/backend/data/final/dataset_binary_visits.csv'
    df = pd.read_csv(dataset_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ==========================================
# CHECK 1: FEATURE-TARGET CORRELATIONS
# ==========================================
print("\n" + "="*70)
print("CHECK 1: FEATURE-TARGET CORRELATIONS")
print("="*70)

X = df.drop(columns=['target', 'date'], errors='ignore')
y = df['target']

# Calculate correlations with target
correlations = pd.DataFrame({
    'feature': X.columns,
    'correlation': [X[col].corr(y) for col in X.columns]
})
correlations = correlations.sort_values('correlation', key=abs, ascending=False)

print("\nTop 10 Features Most Correlated with Target:")
print("-" * 50)
for idx, row in correlations.head(10).iterrows():
    print(f"{row['feature']:40s}: {row['correlation']:+.4f}")

# Flag suspicious correlations
suspicious = correlations[correlations['correlation'].abs() > 0.95]
if len(suspicious) > 0:
    print("\n⚠️  SUSPICIOUS FEATURES (|correlation| > 0.95):")
    for idx, row in suspicious.iterrows():
        print(f"   - {row['feature']}: {row['correlation']:+.4f}")
        print(f"     This feature might be causing data leakage!")

# ==========================================
# CHECK 2: FEATURE IMPORTANCE ANALYSIS
# ==========================================
print("\n" + "="*70)
print("CHECK 2: FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Train a simple model to check feature importance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
rf.fit(X_train, y_train)

# Get feature importances
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print("-" * 50)
for idx, row in importances.head(10).iterrows():
    print(f"{row['feature']:40s}: {row['importance']:.4f}")

# Check if one feature dominates
if importances.iloc[0]['importance'] > 0.5:
    print(f"\n⚠️  WARNING: Feature '{importances.iloc[0]['feature']}' has {importances.iloc[0]['importance']:.1%} importance!")
    print("   This suggests potential data leakage through this feature.")

# ==========================================
# CHECK 3: SYMPTOM-RELATED FEATURES
# ==========================================
print("\n" + "="*70)
print("CHECK 3: CHECKING SYMPTOM-RELATED FEATURES")
print("="*70)

# Check if symptom features are present when target is 0 (no visits)
symptom_features = [col for col in X.columns if any(x in col.lower() for x in 
                   ['symptom', 'respiratory', 'digestive', 'pain', 'fever', 
                    'neurological', 'dermatological', 'cardiovascular', 'urological'])]

print(f"\nFound {len(symptom_features)} symptom-related features")

# Check logical consistency
no_visit_mask = y == 0
if no_visit_mask.sum() > 0:
    for feature in symptom_features[:5]:  # Check first 5
        if feature in X.columns:
            non_zero_when_no_visits = (X.loc[no_visit_mask, feature] != 0).sum()
            if non_zero_when_no_visits > 0:
                print(f"⚠️  {feature}: has {non_zero_when_no_visits} non-zero values when target=0 (no visits)")
                print(f"   This is illogical - no visits should mean no symptoms!")

# ==========================================
# CHECK 4: TEMPORAL LEAKAGE
# ==========================================
print("\n" + "="*70)
print("CHECK 4: CHECKING FOR TEMPORAL LEAKAGE")
print("="*70)

# Check lag features
lag_features = [col for col in X.columns if 'lag' in col.lower()]
rolling_features = [col for col in X.columns if 'rolling' in col.lower()]

print(f"Lag features: {len(lag_features)}")
print(f"Rolling features: {len(rolling_features)}")

# Check if current visit count is in features
if 'visit_count' in X.columns:
    print("\n⚠️  CRITICAL: 'visit_count' is in features!")
    print("   This is the same as the target for binary classification!")

# Check for same-day features that shouldn't be known
current_day_symptoms = [col for col in X.columns if any(x in col for x in 
                        ['respiratory_count', 'digestive_count', 'pain_count', 'fever_count',
                         'total_symptom_load', 'symptom_diversity']) 
                        and 'lag' not in col and 'rolling' not in col]

if current_day_symptoms:
    print(f"\n⚠️  CRITICAL: Found {len(current_day_symptoms)} current-day symptom features:")
    for feat in current_day_symptoms[:5]:
        print(f"   - {feat}")
    print("   These are only known AFTER visits occur - causing data leakage!")

# ==========================================
# CHECK 5: REMOVE LEAKY FEATURES AND RETEST
# ==========================================
print("\n" + "="*70)
print("CHECK 5: TESTING WITHOUT SUSPICIOUS FEATURES")
print("="*70)

# Identify features to remove
features_to_remove = []

# Remove highly correlated features
features_to_remove.extend(suspicious['feature'].tolist())

# Remove current-day symptom counts (not lagged)
features_to_remove.extend(current_day_symptoms)

# Remove visit_count if present
if 'visit_count' in X.columns:
    features_to_remove.append('visit_count')

# Remove features with illogical values
features_to_remove.extend(['total_symptom_load', 'symptom_diversity', 
                          'respiratory_dominance', 'cold_weather_stress'])

# Remove duplicates
features_to_remove = list(set(features_to_remove))

print(f"\nRemoving {len(features_to_remove)} suspicious features:")
for feat in features_to_remove:
    if feat in X.columns:
        print(f"   - {feat}")

# Create clean dataset
X_clean = X.drop(columns=[f for f in features_to_remove if f in X.columns])

print(f"\nOriginal features: {X.shape[1]}")
print(f"Clean features: {X_clean.shape[1]}")

# Retrain and test
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y, test_size=0.2, random_state=42, stratify=y)

rf_clean = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clean.fit(X_train_clean, y_train_clean)

y_pred_original = rf.predict(X_test)
y_pred_clean = rf_clean.predict(X_test_clean)

acc_original = accuracy_score(y_test, y_pred_original)
acc_clean = accuracy_score(y_test_clean, y_pred_clean)

print(f"\n📊 RESULTS:")
print(f"   Accuracy with ALL features: {acc_original:.4f}")
print(f"   Accuracy with CLEAN features: {acc_clean:.4f}")
print(f"   Difference: {acc_original - acc_clean:.4f}")

if acc_original == 1.0 and acc_clean < 0.95:
    print("\n✅ DATA LEAKAGE CONFIRMED!")
    print("   The perfect accuracy was due to leaky features.")
    print("   After removing them, accuracy is more realistic.")

# ==========================================
# RECOMMENDATIONS
# ==========================================
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("""
📋 TO FIX THE PERFECT ACCURACY ISSUE:

1. REMOVE CURRENT-DAY SYMPTOM FEATURES:
   - Use only lagged symptom features (lag_1, lag_3, etc.)
   - Current symptoms are only known AFTER visits occur
   
2. REMOVE DERIVED FEATURES FROM TARGET:
   - total_symptom_load (directly related to visits)
   - symptom_diversity (only non-zero when visits occur)
   - respiratory_dominance (undefined when no symptoms)
   
3. USE ONLY PREDICTIVE FEATURES:
   ✅ Keep: Environmental data (temperature, humidity, pollution)
   ✅ Keep: Temporal features (day_of_week, academic_period)
   ✅ Keep: Lagged features (previous days' patterns)
   ❌ Remove: Same-day symptom counts
   ❌ Remove: Features derived from current visits
   
4. REBUILD THE DATASET:
   - Go back to dataset_optimization.py
   - Ensure target (has_visits) is created from visit_count
   - Then remove visit_count and current-day symptoms from features
   
5. VALIDATE LOGIC:
   - When target=0 (no visits), all symptom features should be 0
   - Features should be things known BEFORE the day's visits
""")

# Save the clean feature list
clean_features = X_clean.columns.tolist()
with open('medisense/backend/data/final/optimized/clean_features.txt', 'w') as f:
    f.write("CLEAN FEATURES (NO DATA LEAKAGE)\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total: {len(clean_features)} features\n\n")
    for feat in clean_features:
        f.write(f"- {feat}\n")

print(f"\n💾 Clean feature list saved to: clean_features.txt")
print("\n🔧 Re-run the pipeline with these features for realistic results!")
