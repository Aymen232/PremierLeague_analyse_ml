# ⚽ Premier League Match Prediction

## 📌 Project Overview

Premier League Match Prediction is an end-to-end Data Engineering and Machine Learning project designed to analyze football matches and predict match outcomes using historical Premier League data.

The project automatically collects real-world football data from an external API, cleans and transforms the data, generates predictive features based on teams' historical performances, trains Machine Learning models, and visualizes insights through interactive Power BI dashboards.

---

## 🎯 Objectives

- Collect Premier League match data from a football REST API.
- Build a robust ETL pipeline for data ingestion and transformation.
- Perform feature engineering using teams' historical performances.
- Train Machine Learning models to predict match outcomes.
- Visualize football statistics and predictions using Power BI.

---

## 🏗️ Project Architecture

```text
API Football
      ↓
Raw Data (CSV)
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Machine Learning Models
      ↓
Predictions & Insights
      ↓
Power BI Dashboard
```

---

## 📂 Project Structure

```text
PremierLeague_analyse_ml/
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/
│
├── powerbi/
│
├── sql/
│
├── src/
│   ├── database/
│   │
│   ├── ingest/
│   │   └── fetch_api_data.py
│   │
│   ├── processing/
│   │   ├── clean_data.py
│   │   └── feature_engineering.py
│   │
│   ├── ml/
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── predict.py
│   │
│   └── utils/
│
├── tests/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

---

## 🔄 Data Pipeline

### 1️⃣ Data Ingestion

Premier League match data is automatically collected from the API-Football REST API.

Run:

```bash
python src/ingest/fetch_api_data.py
```

Output:

```text
data/raw/matches.csv
```

---

### 2️⃣ Data Cleaning

The raw dataset is cleaned and transformed:

- Remove missing values.
- Keep only finished matches.
- Convert dates into datetime format.
- Create the target variable (`result`).

Run:

```bash
python src/processing/clean_data.py
```

Output:

```text
data/processed/matches_cleaned.csv
```

---

### 3️⃣ Feature Engineering

Historical team statistics are generated using the last five matches:

- Average goals scored.
- Average goals conceded.
- Average points earned.

Generated features:

- `home_avg_goals_last5`
- `away_avg_goals_last5`
- `home_avg_conceded_last5`
- `away_avg_conceded_last5`
- `home_points_form_last5`
- `away_points_form_last5`

Run:

```bash
python src/processing/feature_engineering.py
```

Output:

```text
data/features/matches_features.csv
```

---

## 🤖 Machine Learning

The project trains classification models to predict match outcomes:

- Home Win
- Away Win
- Draw

Implemented models:

- Logistic Regression
- Random Forest

Future models:

- XGBoost
- Gradient Boosting

Run:

```bash
python src/ml/train_model.py
```

Current evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score

---

## 📊 Power BI Dashboard

Interactive dashboards are developed using Power BI to visualize:

- Team performances.
- Goals statistics.
- Match results.
- Historical trends.
- Model predictions.

Dashboard pages:

1. League Overview
2. Team Analysis
3. Player Analysis
4. Match Predictions

---

## 🛠️ Technologies Used

### Data Engineering

- Python
- Pandas
- REST API
- Git
- GitHub

### Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest

### Business Intelligence

- Power BI

### Future Integrations

- PostgreSQL
- Docker
- Airflow
- XGBoost
- FastAPI

---

## 🚀 Future Improvements

- Store data in PostgreSQL.
- Automate data ingestion using Apache Airflow.
- Containerize the project with Docker.
- Deploy a prediction API using FastAPI.
- Integrate advanced Machine Learning models such as XGBoost.
- Add CI/CD pipelines using GitHub Actions.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Aymen232/PremierLeague_analyse_ml.git
```

Move into the project directory:

```bash
cd PremierLeague_analyse_ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\Activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file at the root of the project:

```env
API_KEY=your_api_key_here
```

The API key can be obtained from:

https://www.api-football.com/

---

## 👨‍💻 Author

**Aymen Halleb**

Engineering Student in Computer Science
