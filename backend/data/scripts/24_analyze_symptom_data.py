"""
Analyze Symptom Dataset for Issues
===================================
This script analyzes the symptom dataset to identify potential issues
affecting model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("SYMPTOM DATASET ANALYSIS")
print("=" * 80)

# Load the dataset
df = pd.read_csv('medisense/backend/data/final/symptom/individual_symptom_dataset.csv')
print(f"\nDataset shape: {df.shape}")

# ============================================================================
# 1. CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n1. CLASS DISTRIBUTION")
print("-" * 40)

class_dist = df['symptom_category'].value_counts()
print("\nClass Distribution:")
for category, count in class_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  {category:25s}: {count:5d} ({percentage:5.1f}%)")

print(f"\nClass imbalance ratio: {class_dist.max() / class_dist.min():.2f}:1")

# ============================================================================
# 2. FEATURE ANALYSIS
# ============================================================================
print("\n2. FEATURE ANALYSIS")
print("-" * 40)

# Check for missing values
missing = df.isna().sum()
if missing.any():
    print("\nColumns with missing values:")
    print(missing[missing > 0])
else:
    print("\nNo missing values found")

# Check feature variance
numerical_cols = df.select_dtypes(include=[np.number]).columns
numerical_cols = [col for col in numerical_cols if col != 'symptom_category_encoded']

print("\nLow variance features (std < 0.1):")
for col in numerical_cols:
    if df[col].std() < 0.1:
        print(f"  {col}: std={df[col].std():.4f}, unique={df[col].nunique()}")

# ============================================================================
# 3. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n3. FEATURE IMPORTANCE (Quick Random Forest)")
print("-" * 40)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X = df.drop(['symptom_category', 'symptom_category_encoded'], axis=1)
y = df['symptom_category_encoded']

# Quick train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle NaN values
for col in X_train.columns:
    if X_train[col].isna().any():
        X_train[col].fillna(X_train[col].median(), inplace=True)
        X_test[col].fillna(X_test[col].median(), inplace=True)

# Train a quick Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Get feature importances
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
for idx, row in importances.head(15).iterrows():
    print(f"  {row['feature']:40s}: {row['importance']:.4f}")

print("\nBottom 10 Least Important Features:")
for idx, row in importances.tail(10).iterrows():
    print(f"  {row['feature']:40s}: {row['importance']:.4f}")

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================
print("\n4. CORRELATION WITH TARGET")
print("-" * 40)

# Calculate correlation with encoded target
correlations = df[numerical_cols].corrwith(df['symptom_category_encoded']).abs().sort_values(ascending=False)

print("\nTop 10 Features Most Correlated with Target:")
for feature, corr in correlations.head(10).items():
    print(f"  {feature:40s}: {corr:.4f}")

# ============================================================================
# 5. TEMPORAL PATTERNS
# ============================================================================
print("\n5. TEMPORAL PATTERNS")
print("-" * 40)

# Check distribution by day of week
dow_dist = pd.crosstab(df['dow'], df['symptom_category'])
print("\nSymptom distribution by day of week:")
print(dow_dist)

# Check distribution by month
month_dist = pd.crosstab(df['month'], df['symptom_category'])
print("\nSymptom distribution by month:")
print(month_dist)

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n6. CREATING VISUALIZATIONS...")
print("-" * 40)

fig = plt.figure(figsize=(20, 12))

# 1. Class distribution
ax1 = plt.subplot(2, 3, 1)
class_dist.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Class Distribution')
ax1.set_xlabel('Symptom Category')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# 2. Feature importance
ax2 = plt.subplot(2, 3, 2)
importances.head(15).plot(x='feature', y='importance', kind='barh', ax=ax2, color='coral')
ax2.set_title('Top 15 Feature Importances')
ax2.set_xlabel('Importance')
ax2.set_ylabel('Feature')

# 3. Correlation heatmap (top features)
ax3 = plt.subplot(2, 3, 3)
top_features = correlations.head(10).index.tolist()
corr_matrix = df[top_features + ['symptom_category_encoded']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax3)
ax3.set_title('Correlation Matrix (Top Features)')

# 4. Symptom by day of week
ax4 = plt.subplot(2, 3, 4)
dow_dist.T.plot(kind='bar', stacked=True, ax=ax4)
ax4.set_title('Symptoms by Day of Week')
ax4.set_xlabel('Day of Week')
ax4.set_ylabel('Count')
ax4.legend(title='Day', bbox_to_anchor=(1.05, 1), loc='upper left')

# 5. Environmental features distribution
ax5 = plt.subplot(2, 3, 5)
env_features = ['temp', 'humidity', 'pm2_5', 'pm10']
for i, feature in enumerate(env_features):
    if feature in df.columns:
        ax5.hist(df[feature], bins=30, alpha=0.5, label=feature)
ax5.set_title('Environmental Features Distribution')
ax5.set_xlabel('Value')
ax5.set_ylabel('Frequency')
ax5.legend()

# 6. Visit patterns
ax6 = plt.subplot(2, 3, 6)
visit_features = ['daily_visits_count', 'visit_lag1', 'visit_lag7']
for feature in visit_features:
    if feature in df.columns:
        ax6.hist(df[feature].dropna(), bins=30, alpha=0.5, label=feature)
ax6.set_title('Visit Pattern Features')
ax6.set_xlabel('Visit Count')
ax6.set_ylabel('Frequency')
ax6.legend()

plt.tight_layout()
plt.savefig('medisense/backend/data/visualization/final/symptom_data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("\nKEY FINDINGS:")
print(f"1. Severe class imbalance: {class_dist.max() / class_dist.min():.2f}:1 ratio")
print(f"2. Smallest class has only {class_dist.min()} samples")
print(f"3. Dataset size: {len(df)} total samples")

print("\nRECOMMENDATIONS:")
print("1. Consider merging rare symptom categories")
print("2. Use class weights or SMOTE for balancing")
print("3. Try simpler models (Logistic Regression) first")
print("4. Consider binary classification (most common vs rest)")
print("5. Collect more data for rare categories")
print("6. Feature engineering: interaction terms, polynomial features")
print("7. Try ensemble methods with class balancing")

print("\n" + "=" * 80)
