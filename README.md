#  Diabetes Prediction System

## 📌 Project Overview

The **Diabetes Prediction System** is a machine learning classification project that predicts whether a patient is **diabetic or not** based on medical and demographic features. The project covers the complete **data science workflow**: data exploration, visualization, preprocessing, model training, hyperparameter tuning, and evaluation.

This project is ideal for demonstrating practical skills in **EDA, feature scaling, model comparison, and classification metrics**.

---

## 🎯 Objectives

* Analyze and understand diabetes-related medical data
* Build reliable classification models
* Handle class imbalance effectively
* Compare multiple machine learning algorithms
* Predict diabetes for new patient data

---

## 📂 Dataset

* **Source:** Pima Indians Diabetes Dataset
* **Records:** 768 patients
* **Features:** 8 medical attributes
* **Target:** `Outcome` (0 = Not Diabetic, 1 = Diabetic)

### 🔢 Features Description

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

---

## 🔍 Exploratory Data Analysis (EDA)

The following steps were performed:

* Data shape, types, and summary statistics
* Missing values and duplicate checks
* Class distribution analysis
* Visualizations:

  * Outcome distribution
  * Boxplots (Glucose, BMI vs Outcome)
  * Correlation heatmap
  * Scatter plot (Glucose vs BMI)

Key insight: **Glucose and BMI show strong correlation with diabetes outcome**.

---

## ⚙️ Data Preprocessing

* Feature-target separation
* Train-test split (80% train, 20% test) with stratification
* Feature scaling using **StandardScaler**
* Handling class imbalance using `class_weight='balanced'`

---

## 🤖 Models Implemented

The following machine learning models were trained and optimized using **GridSearchCV**:

### 1️⃣ Logistic Regression

* Regularization: L1 & L2
* Best CV Accuracy: ~75%

### 2️⃣ Random Forest Classifier ⭐

* Best performing model
* Tuned hyperparameters:

  * Number of trees
  * Depth
  * Class weights
* Best Test Accuracy: **~76%**

### 3️⃣ Support Vector Machine (SVM)

* RBF & Linear kernels
* Optimized for F1-score
* High recall for diabetic cases

---

## 📊 Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The **Random Forest Classifier** achieved the best balance between precision and recall.

---

## 🧪 Sample Prediction

```python
new_patient = [2, 120, 70, 20, 79, 25.0, 0.351, 35]
result = predict_diabetes(new_patient, best_rf, scaler)
```

**Output:** `Not Diabetic`

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**

  * NumPy, Pandas
  * Matplotlib, Seaborn
  * Scikit-learn
* **Environment:** Google Colab / Jupyter Notebook

---

## 📁 Project Structure

```
Diabetes-Prediction-System/
│
├── diabetes.csv
├── diabetes_prediction.ipynb
├── README.md
```

---

