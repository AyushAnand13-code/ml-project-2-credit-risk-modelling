# 🧠 Credit Risk Prediction App

A machine learning web app built using **Streamlit** that predicts the likelihood of a loan default based on applicant information. The backend model is trained using **Logistic Regression** and **XGBoost**, with thorough preprocessing and feature engineering.

---

## 📌 Project Overview

Credit risk assessment is crucial for financial institutions to evaluate the probability of a borrower defaulting on a loan. This project includes:

- End-to-end machine learning pipeline
- Preprocessing of categorical and numerical features
- Model training using Logistic Regression and XGBoost
- Deployment via Streamlit with real-time prediction

---

## 🚀 Features

- 📊 **Two ML models**: Logistic Regression (interpretable) & XGBoost (robust performance)
- 🧹 **Data Preprocessing**: Label Encoding, Scaling, Null handling
- 🧠 **Interactive UI**: Takes user input and returns prediction


---

## 🧰 Tech Stack

| Tool           | Description                      |
|----------------|----------------------------------|
| Python         | Programming Language             |
| Pandas, NumPy  | Data manipulation and cleaning   |
| scikit-learn   | ML models, pipeline, preprocessing |
| XGBoost        | Advanced gradient boosting model |
| Streamlit      | Web app frontend                 |
| joblib         | Model serialization              |


## 🚀 How to Run the Project Locally

Follow the steps below to run the credit risk prediction project on your local machine.

### 1. Clone the Repository


git clone https://github.com/AyushAnand13-code/ml-project-2-credit-risk-modelling.git
cd ml-project-2-credit-risk-modelling


### 2.Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3.Install Required Dependencies
pip install -r requirements.txt

### 4. Run the Streamlit App
streamlit run app/app.py



