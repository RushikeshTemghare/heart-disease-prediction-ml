# Heart Disease Prediction using Machine Learning

An end-to-end Machine Learning project that predicts the likelihood of heart disease using multiple supervised classification algorithms. The project follows a complete data science workflow including Exploratory Data Analysis (EDA), data preprocessing, feature engineering, model training, evaluation, and deployment of the best-performing model using a Streamlit web application.

![Python](https://img.shields.io/badge/Python-blue)
![Pandas](https://img.shields.io/badge/Pandas-150458)
![NumPy](https://img.shields.io/badge/NumPy-013243)
![Matplotlib](https://img.shields.io/badge/Matplotlib-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B)
![Joblib](https://img.shields.io/badge/Joblib-3776AB)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-success)
![Classification](https://img.shields.io/badge/Classification-Binary-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📖 Project Overview

Heart disease remains one of the leading causes of death worldwide. Early prediction of cardiovascular disease can assist healthcare professionals in identifying high-risk patients and supporting timely medical intervention.

This project demonstrates a complete end-to-end supervised machine learning pipeline for predicting whether a patient is likely to have heart disease based on several clinical attributes.

The project includes every stage of a real-world machine learning workflow, from understanding the dataset and preparing the data to comparing multiple machine learning algorithms and deploying the final trained model as an interactive web application using Streamlit.

The deployed application enables users to enter patient information and instantly receive a prediction indicating whether the patient is at a higher or lower risk of heart disease.

---

# 🎯 Problem Statement

The objective of this project is to develop a supervised machine learning classification model capable of predicting the presence of heart disease using patient clinical information.

Medical professionals collect numerous cardiovascular indicators during routine examinations. By analysing these attributes using machine learning techniques, predictive models can assist in identifying patients who may require further clinical assessment.

This project aims to:

- Build an accurate heart disease prediction model.
- Compare multiple supervised machine learning algorithms.
- Select the best-performing model based on evaluation metrics.
- Deploy the final model using Streamlit for real-time prediction.

---

# 📊 Dataset

This project uses the **Heart Disease Dataset** stored as a CSV file (`heart.csv`).

The dataset contains patient medical records collected from multiple clinical sources and includes both numerical and categorical features commonly used for cardiovascular disease diagnosis.

### Dataset Features

| Feature | Description |
|----------|-------------|
| Age | Age of the patient |
| Sex | Patient gender |
| ChestPainType | Type of chest pain experienced |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dL) |
| FastingBS | Fasting blood sugar greater than 120 mg/dL |
| RestingECG | Resting electrocardiogram results |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina |
| Oldpeak | ST depression induced by exercise relative to rest |
| ST_Slope | Slope of the peak exercise ST segment |
| HeartDisease | Target variable indicating presence (1) or absence (0) of heart disease |

The dataset contains a combination of categorical and numerical variables, making it suitable for supervised binary classification.

---

# 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was performed to better understand the dataset before model development.

The notebook includes:

- Dataset inspection
- Shape and dimensionality analysis
- Data type verification
- Summary statistics
- Missing value analysis
- Feature distribution analysis
- Correlation analysis
- Identification of categorical and numerical variables

Data visualisation techniques were used to explore feature distributions and understand relationships between variables using **Matplotlib** and **Seaborn**.

The insights obtained during EDA helped guide the preprocessing and feature engineering steps prior to model training.

---

# 🧹 Data Preprocessing

Before training the machine learning models, the dataset underwent several preprocessing steps to improve model performance and ensure consistency.

The preprocessing pipeline included:

- Checking for missing values
- Removing unnecessary inconsistencies
- Separating categorical and numerical features
- Encoding categorical variables using One-Hot Encoding
- Scaling numerical features using **StandardScaler**
- Preparing the final feature matrix for model training

The fitted scaler was saved using **Joblib** and later reused during deployment to ensure that user input received the same preprocessing as the training data.

---

# ⚙️ Feature Engineering

Feature engineering focused on transforming categorical variables into a machine learning-compatible format.

The following techniques were applied:

- One-Hot Encoding of categorical features
- Creation of the final feature matrix
- Preservation of feature order for deployment
- Saving the expected feature columns for use in the Streamlit application

This ensures that user input is transformed into exactly the same format as the data used during model training.

---

# ⚙️ Machine Learning Workflow

This project follows a complete supervised machine learning pipeline from raw data to deployment.

The workflow includes:

1. Import Required Libraries
2. Load the Dataset
3. Exploratory Data Analysis (EDA)
4. Data Cleaning
5. Feature Engineering
6. One-Hot Encoding of Categorical Features
7. Feature Scaling using StandardScaler
8. Train-Test Split
9. Model Training
10. Model Prediction
11. Model Evaluation
12. Model Comparison
13. Selection of the Best Performing Model
14. Saving the Trained Model using Joblib
15. Deployment using Streamlit

This structured workflow follows industry-standard machine learning practices and ensures that the deployed application uses the same preprocessing pipeline as the training dataset.

---

# 🤖 Machine Learning Models Implemented

Several supervised machine learning classification algorithms were trained and evaluated to identify the best-performing model.

The models implemented include:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree Classifier

Each model was trained using the same preprocessed dataset and evaluated using common classification metrics to ensure a fair comparison.

---

# 📈 Model Evaluation

The performance of each machine learning model was evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score

The comparison of the models is shown below.

| Model | Accuracy | F1 Score |
|--------|----------:|----------:|
| **K-Nearest Neighbors (KNN)** | **88.59%** | **89.86%** |
| Logistic Regression | 87.50% | 88.78% |
| Naive Bayes | 86.96% | 87.88% |
| Support Vector Machine (SVM) | 86.41% | 88.04% |
| Decision Tree Classifier | 75.54% | 77.16% |

---

# 🏆 Final Model Selection

After comparing the performance of multiple classification algorithms, the **K-Nearest Neighbors (KNN)** classifier achieved the highest overall performance.

The KNN model achieved:

- ✅ Accuracy: **88.59%**
- ✅ F1 Score: **89.86%**

Due to its superior predictive performance, the KNN model was selected as the final model for deployment.

The trained model, feature scaler, and feature column structure were saved using **Joblib** for use within the Streamlit application.

Saved model files:

- `knn_heart_model.pkl`
- `heart_scaler.pkl`
- `heart_columns.pkl`

---

# 🚀 Streamlit Web Application

The final K-Nearest Neighbors (KNN) model was deployed using **Streamlit**, providing an interactive web interface that allows users to obtain real-time heart disease predictions.

The application loads the trained machine learning model along with the preprocessing components and generates predictions based on user-provided clinical information.

The deployed application ensures that incoming user data undergoes the same preprocessing pipeline used during model training, maintaining prediction consistency.

---

# 🖥️ Application Features

The Streamlit application provides an intuitive interface where users can enter the following patient information:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak (ST Depression)
- ST Slope

The application then:

- Creates a structured input dataframe
- Performs one-hot encoding alignment
- Applies the saved StandardScaler
- Generates predictions using the trained KNN model
- Displays whether the patient has a **High Risk** or **Low Risk** of Heart Disease

The application also includes:

- Professional user interface
- Medical disclaimer
- Interactive widgets
- Real-time prediction
- Clean and responsive layout

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Development Environment

- Jupyter Notebook
- Visual Studio Code

### Data Analysis

- Pandas
- NumPy

### Data Visualisation

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Model Persistence

- Joblib

### Web Application

- Streamlit

---

# 📂 Model Files

The trained machine learning artifacts are stored using Joblib.

| File | Purpose |
|------|----------|
| `knn_heart_model.pkl` | Trained K-Nearest Neighbors classification model |
| `heart_scaler.pkl` | StandardScaler used during preprocessing |
| `heart_columns.pkl` | Feature column structure used during prediction |

These files ensure that the deployed application performs preprocessing and prediction exactly as carried out during model training.

---

# 📂 Repository Structure

```text
heart-disease-prediction-ml/
│
├── heart_disease_prediction.ipynb      # Complete machine learning workflow
├── app.py                              # Streamlit web application
├── heart.csv                           # Dataset
├── knn_heart_model.pkl                 # Trained KNN model
├── heart_scaler.pkl                    # StandardScaler object
├── heart_columns.pkl                   # Feature column structure
├── README.md                           # Project documentation
├── LICENSE                             # MIT License
└── .gitignore                          # Git ignore file
```

---

# ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
```

Navigate into the project directory:

```bash
cd heart-disease-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

Once the server starts, open your browser and navigate to:

```
http://localhost:8501
```

The application will load the trained machine learning model and allow users to perform heart disease risk predictions through an interactive web interface.

---

# 📸 Application Preview

The Streamlit application provides:

- ❤️ Professional dashboard interface
- 🩺 Patient information form
- 📊 Real-time heart disease prediction
- ⚠️ High-risk notification
- ✅ Low-risk notification
- 📋 Medical disclaimer
- 🤖 Machine learning powered prediction

> **Note:** Screenshots of the application can be added here after deployment.

---

# 💡 Skills Demonstrated

This project demonstrates practical experience in:

- Python Programming
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Data Preprocessing
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Supervised Machine Learning
- Binary Classification
- Model Comparison
- Model Evaluation
- Model Selection
- Model Persistence using Joblib
- Streamlit Application Development
- Machine Learning Deployment
- Git & GitHub Version Control

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Building an end-to-end machine learning pipeline from raw data to deployment.
- Understanding the importance of exploratory data analysis before model development.
- Preparing datasets through preprocessing and feature engineering techniques.
- Implementing and comparing multiple supervised classification algorithms.
- Evaluating machine learning models using Accuracy, Precision, Recall and F1 Score.
- Selecting the most suitable model based on performance metrics.
- Saving trained machine learning models for deployment using Joblib.
- Building an interactive web application using Streamlit.
- Deploying machine learning models for real-time predictions.
- Following software engineering best practices for reproducible machine learning projects.

---

# 🚀 Future Improvements

Potential enhancements for this project include:

- Hyperparameter tuning using GridSearchCV
- Cross-validation for improved model robustness
- Feature importance analysis
- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier
- ROC Curve Analysis
- Precision-Recall Curve Analysis
- Model explainability using SHAP
- Docker containerisation
- Cloud deployment using Streamlit Community Cloud
- User authentication for secure access
- Integration with healthcare databases
- Mobile-friendly interface improvements

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

---

# 👤 Author

**Rushikesh Temghare**

MSc Data Science & Artificial Intelligence  
Bournemouth University

Passionate about Machine Learning, Artificial Intelligence, Data Analytics, and Data Science.

---

# ⚠️ Disclaimer

This application has been developed for educational and portfolio purposes only.

The predictions generated by this machine learning model should **not** be interpreted as professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional regarding medical conditions or healthcare decisions.

---

# 📄 License

This project is licensed under the **MIT License**.

---

⭐ If you found this project helpful, consider giving the repository a **star**.
