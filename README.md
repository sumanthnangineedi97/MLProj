# 🧠 Student Exam Performance Predictor

This is a Machine Learning web application that predicts a student's **math exam score** based on their demographic and academic details. The project uses a trained regression model and provides predictions via a Flask-based web interface.

---

## 📁 Project Structure
```
.
├── app.py
├── artifacts
│   ├── mlflow_model.pkl
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── test.csv
│   └── train.csv
├── Dockerfile
├── mlartifacts
│   └── 0
├── mlruns
│   ├── 0
│   └── models
├── model_train_pipeline.py
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── components
│   ├── exception.py
│   ├── logger.py
│   ├── pipeline
│   └── utils.py
└── templates
    └── home.html
```

## 🚀 Features

- Predicts **Math Exam Score** using:
  - Gender
  - Ethnicity
  - Parental education
  - Lunch type
  - Test prep course
  - Reading & Writing scores
- Model training pipeline with modular code
- MLflow tracking enabled
- Web interface using Flask
- Dockerized for easy deployment

---
## 📽 Flask Website Output

[![Click to watch demo](./assets/demo-thumb.png)](https://github.com/user-attachments/assets/6ec086a5-3aca-485f-9759-fdea836b0dd3)

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/mlproj.git
cd mlproj
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
## 🧪 Train the Model
To train the regression model and generate artifacts:
```bash
python model_train_pipeline.py
```

## 🌐 Run the Web App
```bash
python app.py
```
## 🐳 Docker Usage
Build the Docker image
```bash
docker build -t student-exam-app .
```
Run the container
```bash
docker run -d -p 5000:5000 student-exam-app
```

## 📊 MLflow Tracking
```bash
mlflow ui
```

## 📌 TODOs
- Add unit tests
- Improve UI styling
- Integrate with a cloud backend (e.g., AWS/GCP)
- Add CI/CD workflow

