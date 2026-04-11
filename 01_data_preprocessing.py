"""
Student Performance Prediction - Clean Data Preprocessing
No external dependencies issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Create directories
for dir_name in ['data', 'models', 'visualizations']:
    os.makedirs(dir_name, exist_ok=True)

print("=" * 80)
print(" STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 80)

# ============================================
# STEP 1: CREATE DATASET
# ============================================
print("\n[1/5] Creating student dataset...")

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
n_students = 1000

# Create synthetic data
data = {
    # Demographics
    'age': np.random.choice([15, 16, 17, 18, 19, 20, 21, 22], n_students, 
                            p=[0.12, 0.20, 0.25, 0.18, 0.12, 0.07, 0.04, 0.02]),
    'gender': np.random.choice(['M', 'F'], n_students, p=[0.48, 0.52]),
    'address': np.random.choice(['Urban', 'Rural'], n_students, p=[0.70, 0.30]),
    
    # Family Background
    'parent_education': np.random.choice([0, 1, 2, 3, 4], n_students, 
                                         p=[0.05, 0.12, 0.28, 0.32, 0.23]),
    'family_support': np.random.choice(['yes', 'no'], n_students, p=[0.55, 0.45]),
    
    # Academic Factors
    'study_time': np.random.choice([1, 2, 3, 4], n_students, 
                                   p=[0.18, 0.32, 0.35, 0.15]),
    'past_failures': np.random.choice([0, 1, 2, 3], n_students, 
                                      p=[0.68, 0.18, 0.10, 0.04]),
    'absences': np.random.randint(0, 45, n_students),
    'school_support': np.random.choice(['yes', 'no'], n_students, p=[0.25, 0.75]),
    'extra_classes': np.random.choice(['yes', 'no'], n_students, p=[0.32, 0.68]),
    'internet': np.random.choice(['yes', 'no'], n_students, p=[0.62, 0.38]),
    'higher_education': np.random.choice(['yes', 'no'], n_students, p=[0.72, 0.28]),
    
    # Social Factors
    'free_time': np.random.randint(1, 6, n_students),
    'going_out': np.random.randint(1, 6, n_students),
    'health': np.random.randint(1, 6, n_students),
    
    # Grades
    'G1': np.zeros(n_students),  # First period
    'G2': np.zeros(n_students),  # Second period
    'G3': np.zeros(n_students)   # Final grade
}

# Create realistic grades
for i in range(n_students):
    # Base from study time (1-4 scale)
    base_grade = data['study_time'][i] * 3.5
    
    # Subtract failures impact (each failure reduces by 2-3 points)
    base_grade -= data['past_failures'][i] * 2.5
    
    # Subtract absences impact (each 10 absences reduces by 1 point)
    base_grade -= data['absences'][i] * 0.1
    
    # Add parent education boost
    base_grade += data['parent_education'][i] * 0.6
    
    # Family support boost
    if data['family_support'][i] == 'yes':
        base_grade += 1.2
    
    # School support boost
    if data['school_support'][i] == 'yes':
        base_grade += 0.8
    
    # Higher education goal boost
    if data['higher_education'][i] == 'yes':
        base_grade += 1.5
    
    # Add randomness
    base_grade += np.random.normal(0, 2)
    
    # Clamp between 0 and 20
    final_grade = max(0, min(20, base_grade))
    
    # Assign grades (G3 is final, G2 and G1 are earlier periods)
    data['G3'][i] = final_grade
    data['G2'][i] = max(0, min(20, final_grade - np.random.randint(0, 3)))
    data['G1'][i] = max(0, min(20, final_grade - np.random.randint(0, 4)))

# Create DataFrame
df = pd.DataFrame(data)

print(f"✓ Created dataset with {len(df)} students")
print(f"✓ Average Age: {df['age'].mean():.1f}")
print(f"✓ Gender Balance: {df['gender'].value_counts().to_dict()}")

# ============================================
# STEP 2: FEATURE ENGINEERING
# ============================================
print("\n[2/5] Feature engineering...")

# Create engagement score
df['engagement_score'] = (df['study_time'] * 2 + 
                          df['extra_classes'].map({'yes': 1, 'no': 0}) * 1.5 +
                          df['internet'].map({'yes': 1, 'no': 0}) * 1)

# Create risk score
df['risk_score'] = (df['past_failures'] * 2 + 
                    df['absences'] / 10 - 
                    df['study_time'] / 2)

# Create performance trend
df['performance_trend'] = df['G3'] - df['G1']

# Create family support score
df['family_score'] = (df['parent_education'] * 0.5 + 
                      df['family_support'].map({'yes': 1, 'no': 0}) * 1.5)

print("✓ Created 4 advanced features")

# ============================================
# STEP 3: CREATE TARGET VARIABLES
# ============================================
print("\n[3/5] Creating target variables...")

# Binary target: Pass (grade >= 10)
df['passed'] = (df['G3'] >= 10).astype(int)

# Performance level categories
df['performance_level'] = pd.cut(df['G3'], 
                                 bins=[-1, 8, 12, 16, 20],
                                 labels=['Poor', 'Average', 'Good', 'Excellent'])

# Improvement needed flag
df['needs_improvement'] = ((df['G3'] < 10) | (df['performance_trend'] < 0)).astype(int)

pass_rate = df['passed'].mean() * 100
print(f"✓ Pass Rate: {pass_rate:.1f}% ({df['passed'].sum()} students)")
print(f"✓ Fail Rate: {(1-df['passed'].mean())*100:.1f}% ({(df['passed']==0).sum()} students)")
print(f"✓ Excellent Students: {(df['G3'] >= 16).mean()*100:.1f}%")
print(f"✓ Needs Improvement: {df['needs_improvement'].mean()*100:.1f}%")

# ============================================
# STEP 4: CREATE VISUALIZATIONS
# ============================================
print("\n[4/5] Creating visualizations...")

# Figure 1: Grade Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grade histogram
axes[0, 0].hist(df['G3'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Pass Threshold')
axes[0, 0].set_xlabel('Final Grade')
axes[0, 0].set_ylabel('Number of Students')
axes[0, 0].set_title('Distribution of Final Grades')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Pass/Fail pie chart
pass_fail = [df['passed'].sum(), len(df) - df['passed'].sum()]
colors = ['#51cf66', '#ff6b6b']
explode = (0.05, 0)
axes[0, 1].pie(pass_fail, labels=['Pass', 'Fail'], autopct='%1.1f%%', 
               colors=colors, startangle=90, explode=explode)
axes[0, 1].set_title('Pass vs Fail Distribution')

# Study time impact
study_data = df.groupby('study_time')['G3'].mean()
axes[1, 0].bar(study_data.index, study_data.values, color='#1E88E5', edgecolor='black')
axes[1, 0].set_xlabel('Study Time Level (1-4)')
axes[1, 0].set_ylabel('Average Grade')
axes[1, 0].set_title('Average Grade by Study Time')
axes[1, 0].set_xticks([1, 2, 3, 4])
axes[1, 0].set_xticklabels(['<2h', '2-5h', '5-10h', '>10h'])
axes[1, 0].grid(True, alpha=0.3)

# Failures impact
fail_data = df.groupby('past_failures')['G3'].mean()
axes[1, 1].bar(fail_data.index, fail_data.values, color='#ff9f43', edgecolor='black')
axes[1, 1].set_xlabel('Number of Past Failures')
axes[1, 1].set_ylabel('Average Grade')
axes[1, 1].set_title('Average Grade by Past Failures')
axes[1, 1].set_xticks([0, 1, 2, 3])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/grade_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Grade analysis visualization saved")

# Figure 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = ['age', 'study_time', 'past_failures', 'absences', 'G1', 'G2', 'G3', 'passed']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu', center=0, square=True)
plt.title('Correlation Matrix - Key Features')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Correlation heatmap saved")

# Figure 3: Box plots by pass/fail
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

features_to_plot = ['study_time', 'past_failures', 'absences', 'G1', 'G2', 'engagement_score']
for idx, feature in enumerate(features_to_plot):
    row = idx // 3
    col = idx % 3
    df.boxplot(column=feature, by='passed', ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Pass/Fail')
    axes[row, col].set_xlabel('')
    axes[row, col].set_xticklabels(['Fail', 'Pass'])

plt.suptitle('Feature Distributions by Student Outcome')
plt.tight_layout()
plt.savefig('visualizations/boxplots.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Box plots saved")

# ============================================
# STEP 5: SAVE DATA
# ============================================
print("\n[5/5] Saving data...")

# Save main dataset
df.to_csv('data/student_data_final.csv', index=False)
print("✓ Main dataset saved to 'data/student_data_final.csv'")

# Save passing and at-risk students
df[df['passed'] == 1].to_csv('data/passing_students.csv', index=False)
df[df['passed'] == 0].to_csv('data/at_risk_students.csv', index=False)
print("✓ Subsets saved to 'data/'")

# Save feature information
feature_info = {
    'n_samples': len(df),
    'n_features': len(df.columns),
    'pass_rate': float(pass_rate),
    'average_grade': float(df['G3'].mean()),
    'features': list(df.columns),
    'categorical_features': list(df.select_dtypes(include=['object']).columns),
    'numerical_features': list(df.select_dtypes(include=['number']).columns)
}

with open('models/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("✓ Feature info saved to 'models/feature_info.json'")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 80)
print("✅ DATA PREPROCESSING COMPLETE!")
print("=" * 80)
print(f"\n📊 Dataset Summary:")
print(f"   Total Students: {len(df)}")
print(f"   Features: {len(df.columns)}")
print(f"   Pass Rate: {pass_rate:.1f}%")
print(f"   Average Grade: {df['G3'].mean():.2f}/20")
print(f"   At-Risk Students: {(df['G3'] < 10).sum()}")
print(f"\n📁 Saved Files:")
print(f"   • data/student_data_final.csv")
print(f"   • data/passing_students.csv")
print(f"   • data/at_risk_students.csv")
print(f"   • models/feature_info.json")
print(f"   • visualizations/grade_analysis.png")
print(f"   • visualizations/correlation_heatmap.png")
print(f"   • visualizations/boxplots.png")
print("\n🚀 Next: Run model_training.py")