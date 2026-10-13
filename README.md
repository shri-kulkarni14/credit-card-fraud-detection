# 💳 Credit Card Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange.svg" alt="Model">
  <img src="https://img.shields.io/badge/Imbalance%20Handling-SMOTE-green.svg" alt="SMOTE">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status">
</p>

---

## 📌 Project Overview
An end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions. Built using a heavily imbalanced Kaggle benchmark dataset containing **284,807 transactions** (where only **0.172%** are fraudulent).

---

## 📊 Dataset Details
* **Source:** Kaggle Credit Card Fraud Detection
* **Total Volume:** 284,807 transactions
* **Fraud Cases:** 492 cases (0.172%)
* **Features:** 28 PCA-transformed numerical features (`V1`–`V28`), alongside original `Time` and `Amount` columns.

---

## 🛠️ Methodology & Architecture
1. **Data Preprocessing:** Utilized `RobustScaler` to scale `Amount` and `Time`, ensuring resilience against extreme outliers.
2. **Train-Test Split:** Performed a stratified split ($80/20$) *before* resampling to prevent data leakage and preserve real-world class distributions.
3. **Handling Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) exclusively on the training set to balance minority classes.
4. **Model Training:** Trained a robust **Random Forest Classifier** (100 estimators, `max_depth=10`).

---

## 📈 Key Results & Performance
* **Precision-Recall AUC:** `0.8126`
* **Fraud Recall:** `0.87` *(Successfully caught 87% of actual frauds)*
* **Fraud Precision:** `0.40`

---

## 🚀 How to Run the Project

Clone the repository and execute the following commands in your terminal:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter Notebook
jupyter notebook fraud_detection.ipynb
