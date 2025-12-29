**Handling Class Imbalance in Churn Prediction**

This project addresses the **class imbalance problem** in telecom customer churn prediction by building a scalable, leakage-safe machine learning pipeline. Multiple resampling techniques and models are compared to improve minority class (churner) detection and deliver actionable business insights.


**Project Overview**
- **Domain:** Telecom customer churn prediction  
- **Dataset:** Public BigML telecom churn dataset  
- **Main Challenge:** Severe class imbalance (churners vs non-churners)  
- **Goal:** Improve churn detection using robust ML + resampling strategies  

---

**Resampling Techniques**
- SMOTE  
- SMOTE + ENN  
- ADASYN  
- Random Over Sampling (ROS)  
- Random Under Sampling (RUS)  

---

**Machine Learning Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- Artificial Neural Network (ANN)  
- XGBoost  

---

**Model Training & Evaluation**
- Hyperparameter tuning with **Optuna**
- **Stratified K-Fold Cross-Validation** (leakage-safe)
- Evaluation metrics:
  - F1-score
  - ROC-AUC
  - PR-AUC
  - Precision & Recall

---

**Key Findings**
- **ROS and SMOTE** consistently delivered the best performance
- **XGBoost and Random Forest** achieved the highest F1 and PR-AUC scores
- Hyperparameter tuning significantly improved results
- Final models were serialized for deployment

---

**Tech Stack**
- Python
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Optuna
- Pandas, NumPy, Matplotlib

---

**Project Structure**
- preprocessing/ # Data cleaning, encoding, scaling
- resampling/ # SMOTE, ROS, ADASYN, etc.
- models/ # ML model implementations
- tuning/ # Optuna hyperparameter tuning
- evaluation/ # Metrics, ROC & PR curves
- models_saved/ # Serialized final models (.pkl)

---

**Business Impact**
Model outputs are translated into **data-driven retention strategies**, enabling telecom companies to proactively identify high risk customers and optimize retention efforts.

---

**Project done by:**
@selidik & @melisalevli
