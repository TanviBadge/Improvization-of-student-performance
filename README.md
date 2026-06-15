# 🎓 Improvization of Student Performance using Machine Learning

The AI that never gives up on a student - predicting performance, providing solutions.

## 📌 Project Overview

This project focuses on the **Improvization of Student Performance** using Machine Learning algorithms. The system predicts whether a student will pass or fail based on academic, demographic, and behavioral factors. It helps educators identify at-risk students early and provides personalized recommendations for improvement.

## 🎯 Problem Statement

Traditional education systems identify struggling students only after exam results, making intervention too late. Teachers cannot monitor every student individually in large classrooms. This project solves this by providing an **AI-powered early warning system**.

## 💡 Solution

Our system:
- ✅ Predicts student performance BEFORE final exams
- ✅ Identifies WHY a student is struggling
- ✅ Provides PERSONALIZED recommendations
- ✅ Achives 92.6% AUC-ROC accuracy

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.14 |
| Machine Learning | Scikit-learn |
| Best Algorithm | Gradient Boosting |
| Web Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib |
| Data Processing | Pandas, NumPy |

## 📊 Dataset

- **Source:** UCI Student Performance Dataset
- **Students:** 1,000 records
- **Features:** 32 attributes
  Technically 33 columns total, but G3 is the target variable. So 32 features + 1 target = 33 total columns.
- **Target:** Pass/Fail (G3 ≥ 10)

### Key Features:
- G1, G2 (Previous grades)
- Study time
- Past failures
- Absences
- Family support
- Parental education

## 🧠 Machine Learning Algorithms Used

| Algorithm | Accuracy | AUC-ROC |
|-----------|----------|---------|
| Logistic Regression | 82.5% | 89.3% |
| Decision Tree | 81.5% | 87.5% |
| Random Forest | 83.5% | 91.2% |
| **Gradient Boosting** | **84.5%** | **92.6%** |

**Best Model: Gradient Boosting**

## 📈 Results

### Model Performance
- **Accuracy:** 84.5%
- **Precision:** 86.1%
- **Recall:** 88.3%
- **F1-Score:** 87.2%
- **AUC-ROC:** 92.6%

### Feature Importance (What affects performance most)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | G2 (Second Period Grade) | 85% |
| 2 | G1 (First Period Grade) | 72% |
| 3 | Study Time | 58% |
| 4 | Past Failures | 52% |
| 5 | Absences | 45% |

### Key Insights
- Students studying >5 hours/week have **35% higher** success rate
- Every **5 absences** reduces final grade by 1 point
- Students with family support are **28% more likely** to pass
