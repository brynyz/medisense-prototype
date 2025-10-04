"""
Feature Reduction Pipeline
==========================
This script performs systematic feature reduction through:
1. Logical pruning of redundant features
2. Correlation matrix analysis
3. VIF-based iterative reduction
4. Final feature set optimization
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🔬 MEDISENSE FEATURE REDUCTION PIPELINE")
print("=" * 70)
print("Systematically reducing features while maintaining predictive power")
print("=" * 70)

# Load the dataset
df = pd.read_csv('medisense/backend/data/final/dataset_regression_visits.csv')
print(f"\n📊 Initial Dataset:")
print(f"   Shape: {df.shape}")
print(f"   Total features: {df.shape[1] - 2}")  # Excluding target and date

# Store original features for comparison
original_features = df.columns.tolist()
original_features.remove('target')
original_features.remove('date')

# ==========================================
# STEP 1: LOGICAL PRUNING ✂️
# ==========================================

print("\n" + "="*70)
print("STEP 1: LOGICAL PRUNING ✂️")
print("="*70)

# Define features to remove based on logical redundancy
redundant_time_features = [
    'day_of_week',      # Keep sin_week, cos_week instead
    'week_of_year',     # Keep temporal encoding
    'year',             # Less useful for prediction
    'month'             # Keep sin_day, cos_day instead
]

redundant_binary_flags = [
    'is_weekend',       # Can be inferred from day_of_week encoding
    'is_rainy_day',     # Can be inferred from precipitation_sum
    'is_hot_day',       # Can be inferred from temperature_2m_max
    'is_humid_day',     # Can be inferred from relative_humidity_2m_mean
    'is_high_pollution' # Can be inferred from pm2_5_mean
]

correlated_raw_features = [
    'pm10_mean',        # Keep pm2_5_mean (more specific for health)
]

# Additional redundant features based on domain knowledge
additional_redundant = [
    'is_break_period',  # Academic period encoding captures this
    'temp_humidity_interaction',  # Redundant with heat_humidity_index
]

# Combine all features to remove
features_to_remove_logical = (redundant_time_features + 
                              redundant_binary_flags + 
                              correlated_raw_features + 
                              additional_redundant)

print("\n📋 Features to Remove (Logical Pruning):")
for i, feature in enumerate(features_to_remove_logical, 1):
    if feature in df.columns:
        reason = ""
        if feature in redundant_time_features:
            reason = "Redundant time feature (cyclical encoding is superior)"
        elif feature in redundant_binary_flags:
            reason = "Binary flag (model can learn threshold from continuous variable)"
        elif feature in correlated_raw_features:
            reason = "Correlated with pm2_5_mean"
        elif feature in additional_redundant:
            reason = "Domain-specific redundancy"
        print(f"   {i:2d}. {feature:30s} - {reason}")

# Remove logically redundant features
df_pruned = df.drop(columns=[f for f in features_to_remove_logical if f in df.columns])
remaining_features = [f for f in df_pruned.columns if f not in ['target', 'date']]

print(f"\n✅ After Logical Pruning:")
print(f"   Features removed: {len([f for f in features_to_remove_logical if f in df.columns])}")
print(f"   Remaining features: {len(remaining_features)}")

# ==========================================
# STEP 2: CORRELATION MATRIX ANALYSIS 🔍
# ==========================================

print("\n" + "="*70)
print("STEP 2: CORRELATION MATRIX ANALYSIS 🔍")
print("="*70)

# Calculate correlation matrix
correlation_matrix = df_pruned[remaining_features].corr()

# Find highly correlated pairs (|correlation| > 0.85)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.85:
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            high_corr_pairs.append((col1, col2, corr_value))

print(f"\n🔍 Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.85):")
print("\n" + "-"*80)
print(f"{'Feature 1':35s} {'Feature 2':35s} {'Correlation':>10s}")
print("-"*80)

# Sort by absolute correlation value
high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

features_to_remove_corr = []
for pair in high_corr_pairs:
    print(f"{pair[0]:35s} {pair[1]:35s} {pair[2]:10.3f}")
    
    # Decision logic for which feature to keep
    # Prioritize: interpretability, causal link, fewer dependencies
    
    # Rolling features vs lag features - keep lag (more interpretable)
    if 'rolling' in pair[0] and 'lag' in pair[1]:
        features_to_remove_corr.append(pair[0])
    elif 'rolling' in pair[1] and 'lag' in pair[0]:
        features_to_remove_corr.append(pair[1])
    
    # Standard deviation features - usually less important
    elif 'std' in pair[0] and 'std' not in pair[1]:
        features_to_remove_corr.append(pair[0])
    elif 'std' in pair[1] and 'std' not in pair[0]:
        features_to_remove_corr.append(pair[1])
    
    # Interaction features - keep simpler base features
    elif 'interaction' in pair[0] and 'interaction' not in pair[1]:
        features_to_remove_corr.append(pair[0])
    elif 'interaction' in pair[1] and 'interaction' not in pair[0]:
        features_to_remove_corr.append(pair[1])

# Remove duplicates
features_to_remove_corr = list(set(features_to_remove_corr))

print(f"\n📋 Recommended Features to Remove (High Correlation):")
for i, feature in enumerate(features_to_remove_corr, 1):
    print(f"   {i:2d}. {feature}")

# User confirmation simulation (in production, this would be interactive)
print("\n⚠️  Proceeding with recommended removals...")
df_corr_reduced = df_pruned.drop(columns=[f for f in features_to_remove_corr if f in df_pruned.columns])
remaining_features = [f for f in df_corr_reduced.columns if f not in ['target', 'date']]

print(f"\n✅ After Correlation Reduction:")
print(f"   Features removed: {len([f for f in features_to_remove_corr if f in df_pruned.columns])}")
print(f"   Remaining features: {len(remaining_features)}")

# ==========================================
# STEP 3: VIF-BASED REDUCTION 📊
# ==========================================

print("\n" + "="*70)
print("STEP 3: VARIANCE INFLATION FACTOR (VIF) REDUCTION 📊")
print("="*70)

def calculate_vif(df, features):
    """Calculate VIF for all features"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    
    # Standardize the features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features
    )
    
    # Calculate VIF for each feature
    vif_values = []
    for i in range(len(features)):
        try:
            vif = variance_inflation_factor(df_scaled.values, i)
            vif_values.append(vif)
        except:
            vif_values.append(np.inf)
    
    vif_data["VIF"] = vif_values
    return vif_data.sort_values('VIF', ascending=False)

# Iterative VIF reduction
features_for_vif = remaining_features.copy()
removed_by_vif = []
iteration = 0
vif_threshold = 10

print(f"\n🔄 Starting iterative VIF reduction (threshold = {vif_threshold})...")

while True:
    iteration += 1
    print(f"\n   Iteration {iteration}:")
    
    # Calculate VIF for current features
    vif_df = calculate_vif(df_corr_reduced, features_for_vif)
    
    # Find feature with highest VIF
    max_vif = vif_df.iloc[0]['VIF']
    max_vif_feature = vif_df.iloc[0]['Feature']
    
    print(f"      Highest VIF: {max_vif_feature} = {max_vif:.2f}")
    
    if max_vif > vif_threshold:
        # Remove feature with highest VIF
        features_for_vif.remove(max_vif_feature)
        removed_by_vif.append(max_vif_feature)
        print(f"      ❌ Removed: {max_vif_feature}")
    else:
        print(f"      ✅ All features have VIF < {vif_threshold}")
        break
    
    if iteration > 50:  # Safety check
        print("      ⚠️  Maximum iterations reached")
        break

print(f"\n📋 Features Removed by VIF:")
for i, feature in enumerate(removed_by_vif, 1):
    print(f"   {i:2d}. {feature}")

# Final feature set
df_final = df_corr_reduced.drop(columns=[f for f in removed_by_vif if f in df_corr_reduced.columns])
final_features = [f for f in df_final.columns if f not in ['target', 'date']]

print(f"\n✅ After VIF Reduction:")
print(f"   Features removed: {len(removed_by_vif)}")
print(f"   Final features: {len(final_features)}")

# ==========================================
# STEP 4: FINAL REPORT 📝
# ==========================================

print("\n" + "="*70)
print("FINAL FEATURE REDUCTION REPORT 📝")
print("="*70)

# Summary of all removed features
all_removed = (
    [f for f in features_to_remove_logical if f in original_features] +
    [f for f in features_to_remove_corr if f in original_features and f not in features_to_remove_logical] +
    [f for f in removed_by_vif if f in original_features and f not in features_to_remove_logical and f not in features_to_remove_corr]
)

print(f"\n📊 REDUCTION SUMMARY:")
print(f"   Original features: {len(original_features)}")
print(f"   Features removed: {len(all_removed)}")
print(f"   Final features: {len(final_features)}")
print(f"   Reduction: {(len(all_removed)/len(original_features))*100:.1f}%")

print(f"\n📋 ALL REMOVED FEATURES ({len(all_removed)} total):")
print("-"*70)
for i, feature in enumerate(sorted(all_removed), 1):
    removal_stage = ""
    if feature in features_to_remove_logical:
        removal_stage = "Logical Pruning"
    elif feature in features_to_remove_corr:
        removal_stage = "Correlation Analysis"
    elif feature in removed_by_vif:
        removal_stage = "VIF Reduction"
    print(f"{i:3d}. {feature:35s} [{removal_stage}]")

print(f"\n✅ FINAL OPTIMIZED FEATURE SET ({len(final_features)} features):")
print("-"*70)

# Categorize final features
temporal_features = [f for f in final_features if any(x in f for x in ['sin_', 'cos_', 'academic', 'days_since', 'is_exam'])]
lag_features = [f for f in final_features if 'lag_' in f]
rolling_features = [f for f in final_features if 'rolling_' in f]
environmental_features = [f for f in final_features if any(x in f for x in ['temperature', 'humidity', 'pm2_5', 'precipitation', 'wind', 'heat', 'pollution'])]
symptom_features = [f for f in final_features if any(x in f for x in ['respiratory', 'digestive', 'pain', 'fever', 'neurological', 'dermatological', 'cardiovascular', 'urological', 'symptom', 'total_symptom', 'dominance'])]
interaction_features = [f for f in final_features if 'interaction' in f or 'risk' in f or 'stress' in f or 'combo' in f]

# Remove overlaps
lag_features = [f for f in lag_features if f not in symptom_features]
rolling_features = [f for f in rolling_features if f not in symptom_features]
environmental_features = [f for f in environmental_features if f not in interaction_features]

print("\n🕐 Temporal Features:")
for f in temporal_features:
    print(f"   • {f}")

print("\n📈 Lag Features:")
for f in lag_features:
    print(f"   • {f}")

print("\n📊 Rolling Features:")
for f in rolling_features:
    print(f"   • {f}")

print("\n🌤️ Environmental Features:")
for f in environmental_features:
    print(f"   • {f}")

print("\n🏥 Symptom Features:")
for f in symptom_features:
    print(f"   • {f}")

print("\n🔗 Interaction Features:")
for f in interaction_features:
    print(f"   • {f}")

# ==========================================
# SAVE OPTIMIZED DATASETS
# ==========================================

print("\n" + "="*70)
print("SAVING OPTIMIZED DATASETS 💾")
print("="*70)

# Save the reduced dataset
output_path = 'medisense/backend/data/final/optimized/'
import os
os.makedirs(output_path, exist_ok=True)

# Save with reduced features
for dataset_type in ['binary_visits', 'multiclass_visits', 'risk_based', 'regression_visits', 
                     'dominant_symptom', 'symptom_severity', 'respiratory_outbreak']:
    try:
        # Load original dataset
        original_path = f'medisense/backend/data/final/dataset_{dataset_type}.csv'
        if os.path.exists(original_path):
            df_original = pd.read_csv(original_path)
            
            # Select only the final features plus target and date
            columns_to_keep = final_features + ['target', 'date']
            columns_available = [c for c in columns_to_keep if c in df_original.columns]
            df_optimized = df_original[columns_available]
            
            # Save optimized version
            optimized_path = f'{output_path}dataset_{dataset_type}_optimized.csv'
            df_optimized.to_csv(optimized_path, index=False)
            print(f"   ✅ Saved: dataset_{dataset_type}_optimized.csv")
    except Exception as e:
        print(f"   ⚠️  Could not process dataset_{dataset_type}: {str(e)}")

# Save feature list for reference
with open(f'{output_path}optimized_features.txt', 'w') as f:
    f.write("OPTIMIZED FEATURE SET\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total Features: {len(final_features)}\n")
    f.write(f"Reduction from Original: {(len(all_removed)/len(original_features))*100:.1f}%\n\n")
    
    f.write("FEATURES BY CATEGORY:\n")
    f.write("-" * 30 + "\n\n")
    
    f.write(f"Temporal Features ({len(temporal_features)}):\n")
    for feature in temporal_features:
        f.write(f"  - {feature}\n")
    
    f.write(f"\nLag Features ({len(lag_features)}):\n")
    for feature in lag_features:
        f.write(f"  - {feature}\n")
    
    f.write(f"\nRolling Features ({len(rolling_features)}):\n")
    for feature in rolling_features:
        f.write(f"  - {feature}\n")
    
    f.write(f"\nEnvironmental Features ({len(environmental_features)}):\n")
    for feature in environmental_features:
        f.write(f"  - {feature}\n")
    
    f.write(f"\nSymptom Features ({len(symptom_features)}):\n")
    for feature in symptom_features:
        f.write(f"  - {feature}\n")
    
    f.write(f"\nInteraction Features ({len(interaction_features)}):\n")
    for feature in interaction_features:
        f.write(f"  - {feature}\n")

print(f"\n📄 Feature list saved to: {output_path}optimized_features.txt")

# ==========================================
# VISUALIZATION
# ==========================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS 📊")
print("="*70)

# Create visualization of feature reduction
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature reduction summary
ax1 = axes[0, 0]
stages = ['Original', 'After\nLogical', 'After\nCorrelation', 'After\nVIF']
counts = [
    len(original_features),
    len(original_features) - len([f for f in features_to_remove_logical if f in original_features]),
    len(original_features) - len([f for f in features_to_remove_logical if f in original_features]) - len([f for f in features_to_remove_corr if f in original_features]),
    len(final_features)
]
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
bars = ax1.bar(stages, counts, color=colors)
ax1.set_ylabel('Number of Features')
ax1.set_title('Feature Reduction Progress')
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom', fontweight='bold')

# 2. Removal by category pie chart
ax2 = axes[0, 1]
removal_categories = {
    'Logical Pruning': len([f for f in features_to_remove_logical if f in original_features]),
    'Correlation': len([f for f in features_to_remove_corr if f in original_features and f not in features_to_remove_logical]),
    'VIF': len([f for f in removed_by_vif if f in original_features and f not in features_to_remove_logical and f not in features_to_remove_corr])
}
if sum(removal_categories.values()) > 0:
    ax2.pie(removal_categories.values(), labels=removal_categories.keys(), autopct='%1.1f%%', colors=colors[1:])
    ax2.set_title('Features Removed by Method')

# 3. Final feature categories
ax3 = axes[1, 0]
feature_categories = {
    'Temporal': len(temporal_features),
    'Lag': len(lag_features),
    'Rolling': len(rolling_features),
    'Environmental': len(environmental_features),
    'Symptom': len(symptom_features),
    'Interaction': len(interaction_features)
}
ax3.barh(list(feature_categories.keys()), list(feature_categories.values()), color='#2ecc71')
ax3.set_xlabel('Number of Features')
ax3.set_title('Final Feature Distribution by Category')
for i, (cat, count) in enumerate(feature_categories.items()):
    ax3.text(count + 0.5, i, str(count), va='center', fontweight='bold')

# 4. Correlation heatmap of final features (sample)
ax4 = axes[1, 1]
if len(final_features) > 0:
    # Sample up to 15 features for visualization
    sample_features = final_features[:min(15, len(final_features))]
    corr_sample = df_final[sample_features].corr()
    sns.heatmap(corr_sample, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                square=True, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Correlation Matrix (Sample of Final Features)')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)

plt.suptitle('Feature Reduction Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_path}feature_reduction_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved visualization: feature_reduction_analysis.png")

# ==========================================
# FINAL RECOMMENDATIONS
# ==========================================

print("\n" + "="*70)
print("RECOMMENDATIONS 💡")
print("="*70)

print(f"""
1. MODEL TRAINING:
   - Use optimized datasets in '{output_path}' for model training
   - Reduced features should improve training speed and reduce overfitting
   - Monitor model performance to ensure no critical information loss

2. FEATURE IMPORTANCE:
   - After training, analyze feature importance scores
   - Consider further reduction if some features show negligible importance

3. CROSS-VALIDATION:
   - Validate that the reduced feature set maintains predictive performance
   - Compare metrics with full feature set as baseline

4. INTERPRETABILITY:
   - The reduced set improves model interpretability
   - Focus on understanding the impact of retained features

5. FUTURE ITERATIONS:
   - Re-run this pipeline periodically as new data becomes available
   - Adjust VIF threshold if needed (currently {vif_threshold})
""")

print("\n✅ FEATURE REDUCTION COMPLETE!")
print(f"   Reduced from {len(original_features)} to {len(final_features)} features")
print(f"   Ready for optimized model training! 🚀")
