#  Student Performance Prediction System

<p align="center">
  <img src="https://github.com/zaibshahzadi/Student-Prediction-Model/blob/main/github%20bannar%203.png" width="100%" />
</p>

<p align="center">
  <h3 align="center">AI-powered Student Performance Prediction using Machine Learning & Streamlit</h3>
</p>

---

##  Project Overview

The **Student Performance Prediction System** is a Machine Learning-based web application that predicts whether a student is likely to **Pass** or **Fail** based on academic and personal information.

The model is trained using **Random Forest Classifier** and deployed through **Streamlit** to provide an interactive and user-friendly interface. Users can enter student details and instantly receive prediction results along with model confidence, performance level, and personalized improvement suggestions.

This project demonstrates the complete Machine Learning workflow, from data preprocessing and model training to deployment.

---

##  Features

-  Predicts student performance (Pass / Fail)
-  Displays prediction confidence percentage
-  Shows student performance level
-  Generates personalized improvement suggestions
-  Interactive confidence gauge chart
-  User-friendly Streamlit interface
-  Fast predictions using a trained Machine Learning model

---

##  Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Plotly |
| Model Saving | Joblib |
| Web Framework | Streamlit |
| Development Tool | VS Code |
| Notebook | Jupyter Notebook |
| Version Control | Git & GitHub |

---

##  Dataset

The model is trained on a student performance dataset containing academic and demographic information.

### Input Features

- Student_ID
- Gender
- Study_Hours_per_Week
- Attendance_Rate
- Past_Exam_Scores
- Parental_Education_Level
- Internet_Access_at_Home
- Extracurricular_Activities
- Final_Exam_Score
- Pass_Fail (Target Variable)

### Target Variable

-  Pass
-  Fail

---

##  Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Encoding
4. Train-Test Split
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Model Saving using Joblib
9. Streamlit Deployment

---

##  Model Information

| Model | Random Forest Classifier |
|--------|--------------------------|
| Problem Type | Binary Classification |
| Prediction | Pass / Fail |
| Output | Prediction + Confidence Score |

---

##  Model Performance Comparison

| Machine Learning Model | Accuracy |
|-------------------------|:--------:|
|  Random Forest | **██████████████████ 88.03%**  Best Model |
|  Gradient Boosting | **█████████████████░ 86.62%** |
|  XGBoost | **█████████████████░ 86.62%** |
|  Decision Tree | **████████████████░░ 84.51%** |
|  K-Nearest Neighbors (KNN) | **███████████████░░░ 80.28%** |
|  Support Vector Machine (SVM) | **██████████████░░░░ 79.58%** |
|  Logistic Regression | **█████████████░░░░░ 78.17%** |
|  Naive Bayes | **████████████░░░░░░ 77.46%** |

 **Random Forest achieved the highest accuracy (88.03%) and was selected as the final model for deployment in the Streamlit web application.**

---
##  Project Structure

```text
Student-Prediction-Model/
│
├── app.py
├── train_model.py
├── student_prediction_model.pkl
├── requirements.txt
├── README.md
├── student_prediction_model.ipynb
└── .streamlit/
    └── config.toml
```

---

##  How to Run the Project

### Clone Repository

```bash
git clone https://github.com/zaibshahzadi/Student-Prediction-Model.git
```

### Move into Project Folder

```bash
cd Student-Prediction-Model
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

##  Application Screenshots

###  Home Page

https://github.com/zaibshahzadi/Student-Prediction-Model/blob/main/home%20page.png

---

###  Pass Prediction

https://github.com/zaibshahzadi/Student-Prediction-Model/blob/main/pass%20prediction.png

---

###  Fail Prediction

https://github.com/zaibshahzadi/Student-Prediction-Model/blob/main/fail%20prediction.png

---

###  Personalized Suggestions

https://github.com/zaibshahzadi/Student-Prediction-Model/blob/main/pass%20prediction.png

https://github.com/zaibshahzadi/Student-Prediction-Model/blob/main/fail%20prediction.png

---

##  Future Improvements

- Deep Learning implementation
- Student Performance Dashboard
- Database Integration
- User Authentication
- Export Prediction Reports
- Mobile-Friendly Interface
- Cloud Deployment
- Performance Analytics

---

##  Author

### **Zaib Shahzadi**

**BS Computer Science Student**

Passionate about **Machine Learning, Data Science, Python, Artificial Intelligence, and Web Development.**

 Lahore, Pakistan

 Email:
**zaibnaveed6@gmail.com**

 LinkedIn:
**https://www.linkedin.com/in/zaib-shahzadi-bba376290/**

---

##  Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

It motivates me to continue building more Machine Learning and AI projects.
