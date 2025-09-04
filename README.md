# Student Performance Prediction 🎓  

## Project Overview  
This project focuses on predicting **student grades** based on their performance in **Math, Physics, and Chemistry** using **Machine Learning models**.  
The goal is to compare **classification algorithms** (Logistic Regression and Random Forest) and evaluate their accuracy in predicting student grades.  

---

## Files in this Repository 📂  
- **student_dataset.csv** → Dataset containing students’ subject scores and grades  
- **student_performance.py** → Python script for training, testing, and evaluating machine learning models  
- **README.md** → Project documentation  

---

## Tech Stack 🛠️  
- **Python**  
- **Pandas, NumPy** → Data manipulation  
- **Matplotlib, Seaborn** → Data visualization  
- **Scikit-learn** → Machine learning models and evaluation  

---

## Workflow 📊  

### 🔹 Data Loading & Preprocessing  
- Load dataset (`student_dataset.csv`)  
- Extract features (**Math, Physics, Chemistry**) and target (**Grade**)  
- Encode categorical labels using **LabelEncoder**  
- Split into **training and testing sets**  

### 🔹 Model Training  
- **Logistic Regression** (`LogisticRegression`)  
- **Random Forest** (`RandomForestClassifier`)  

### 🔹 Model Evaluation  
- Accuracy Score  
- Classification Report  
- Confusion Matrix Visualization  
- Feature Importance Plot (Random Forest)  

---

## Results ✅  
- **Logistic Regression Accuracy**: ~ (depends on dataset)  
- **Random Forest Accuracy**: ~ (depends on dataset, usually higher)  
- **Visualizations**:  
  - Confusion Matrix for Random Forest predictions  
  - Feature importance bar chart  

---

## Dataset 📂  
The dataset (**student_dataset.csv**) contains:  
- **Math** → Student’s marks in Math  
- **Physics** → Student’s marks in Physics  
- **Chemistry** → Student’s marks in Chemistry  
- **Grade** → Final grade/class label (target variable)  

---

## Contributing 🤝  
Contributions are welcome!  
Feel free to:  
- Fork the repo  
- Suggest improvements  
- Add new models  
