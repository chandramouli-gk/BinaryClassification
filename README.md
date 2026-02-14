# Machine Learning Assignment 2 - Binary Classification Model Comparison

**Author:** ChandraMouli GK  
**BITS ID:** 2025AA05418  
**Date:** February 2026

---

## a. Problem Statement

The objective of this assignment is to develop and compare multiple machine learning models for **binary classification of heart disease**. The task involves predicting whether a patient has heart disease(target) (1) or not (0) based on various clinical and demographic features. This is a supervised learning problem where we aim to:

1. Build and train six different classification models
2. Evaluate their performance using multiple metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC)
3. Compare the models to identify the best performing algorithm
4. Handle class imbalance appropriately
5. Deploy the models in a user-friendly web application for real-time predictions

The solution includes proper data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## b. Dataset Description

### Dataset Overview
- **Source:** Heart Disease Dataset from Kaggle ([Link](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset))
- **Total Records:** 1,025 patients
- **Features:** 13 clinical and demographic features
- **Target Variable:** Binary (0 = No heart disease, 1 = Heart disease present)
- **Data Split:** 80% Training (820 records) / 20% Testing (205 records)

### Feature Descriptions

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| **age** | Numeric | Age of the patient in years | Continuous |
| **sex** | Categorical | Biological sex | 0 = Female, 1 = Male |
| **cp** | Categorical | Chest pain type | 0-3 (typical angina, atypical angina, non-anginal pain, asymptomatic) |
| **trestbps** | Numeric | Resting blood pressure (mm Hg) | Continuous |
| **chol** | Numeric | Serum cholesterol (mg/dL) | Continuous |
| **fbs** | Categorical | Fasting blood sugar > 120 mg/dL | 0 = False, 1 = True |
| **restecg** | Categorical | Resting electrocardiographic results | 0-2 |
| **thalach** | Numeric | Maximum heart rate achieved | Continuous |
| **exang** | Categorical | Exercise induced angina | 0 = No, 1 = Yes |
| **oldpeak** | Numeric | ST depression induced by exercise relative to rest | Continuous |
| **slope** | Categorical | Slope of the peak exercise ST segment | 0-2 |
| **ca** | Categorical | Number of major vessels colored by fluoroscopy | 0-3 |
| **thal** | Categorical | Thalassemia status | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect |
| **target** | Binary | Heart disease presence (Target variable) | 0 = No disease, 1 = Disease |

### Data Preprocessing
- **Missing Values:** Handled using median imputation for numeric features
- **Feature Scaling:** StandardScaler applied to numeric features
- **Categorical Encoding:** One-hot encoding with drop='first' to avoid multicollinearity
- **Class Imbalance:** Addressed using class_weight='balanced' and scale_pos_weight parameters

---

## c. Models Used

Six machine learning models were trained and evaluated on the heart disease dataset:

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| **Logistic Regression** | 0.8195 | 0.9081 | 0.7797 | 0.8932 | 0.8326 | 0.6457 |
| **Decision Tree** | 0.9854 | 0.9854 | 1.0000 | 0.9709 | 0.9852 | 0.9712 |
| **kNN** | 0.7902 | 0.9295 | 0.7632 | 0.8447 | 0.8018 | 0.5836 |
| **Naive Bayes** | 0.8049 | 0.8423 | 0.7692 | 0.8738 | 0.8182 | 0.6153 |
| **Random Forest (Ensemble)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost (Ensemble)** | 0.9854 | 1.0000 | 1.0000 | 0.9709 | 0.9852 | 0.9712 |

## Model Performance Comparison

| ML Model | Performance Summary |
|-----------|----------------------|
| **Logistic Regression** | Achieved **81.95% accuracy** with strong **AUC (0.9081)**, indicating good class discrimination. Precision (77.97%) and recall (89.32%) are fairly balanced, making it reliable for general predictions. As a linear model, it works well when class relationships are linear. The **MCC (0.6457)** suggests moderate overall performance with limited ability to capture complex patterns. |
| **Decision Tree** | Delivered **98.54% accuracy** with near-perfect metrics. Achieved **perfect precision (1.0)** and high recall (97.09%), meaning almost no false positives and very few false negatives. The **MCC (0.9712)** confirms strong predictive power. However, there is a potential risk of overfitting, so proper validation is required before deployment. |
| **k-Nearest Neighbors (kNN)** | Obtained **79.02% accuracy** with good **AUC (0.9295)**. Precision (76.32%) and recall (84.47%) are reasonably balanced. The lower **MCC (0.5836)** indicates moderate correlation with actual labels. Performance depends heavily on feature scaling and choice of k, and further tuning may improve results. |
| **Naive Bayes** | Achieved **80.49% accuracy** with moderate **AUC (0.8423)**. Precision (76.92%) and recall (87.38%) show decent performance in identifying positive cases. The **MCC (0.6153)** indicates moderate overall reliability. The independence assumption may limit effectiveness in medical datasets where features are correlated. |
| **Random Forest (Ensemble)** | Achieved **100% accuracy** with perfect scores across all evaluation metrics, including **MCC (1.0)**. The ensemble of 200 trees with balanced class weights captures complex feature interactions effectively. However, perfect scores may indicate overfitting, and external validation is recommended before production use. Provides useful feature importance insights. |
| **XGBoost (Ensemble)** | Achieved **98.54% accuracy** with perfect **AUC (1.0)** and precision (1.0). High recall (97.09%) and **MCC (0.9712)** demonstrate strong predictive performance. Built-in regularization reduces overfitting compared to Random Forest. The `scale_pos_weight` parameter effectively handles class imbalance, making it a strong candidate for deployment. |                   
```
**End of README**
