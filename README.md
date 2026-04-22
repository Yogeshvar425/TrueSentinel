# TrueSentinel: Full-Stack Machine Learning Pipeline

A sophisticated, full-stack Machine Learning application built entirely from scratch. TrueSentinel bridges the gap between raw web data extraction and complex Natural Language Processing, acting as a complete pipeline from Data Harvesting to Model Training and Web Deployment.

## 🚀 Features

- **From-Scratch ML Pipeline**: Instead of relying on pre-trained black-box APIs, this project fetches, labels, feature-engineers, and trains its own Machine Learning models (Logistic Regression & SVM).
- **Automated Data Harvesting**: Connects natively via the YouTube Data API to fetch live, real-world comments across diverse videos.
- **Smart Lexicon Labeling**: Utilizes a robust NLP lexical ruleset (including Catching Double-Negatives like "not bad") to automatically establish ground-truth labels.
- **Model Training & Comparison**: Vectorizes text using TF-IDF (supporting Bigrams) and strictly trains both Linear Regression and Support Vector Machines, factoring in Class Weight Balancing for robust models that can handle skewed datasets.
- **Live Interactive Dashboard**: An aesthetically pleasing, glassmorphism-styled dashboard built on FastAPI that feeds user inputs through the custom pickled `.pkl` models for real-time sentiment analysis!

## ⚙️ Setup

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your YouTube Data API v3 Key:
   ```env
   YOUTUBE_API_KEY=your_api_key_here
   ```

## 🧠 Training the Model

Before starting the server, you must train your own models:
```bash
python train.py
```
This script will locally harvest YouTube data, generate `custom_training_data.csv`, train the models, report Accuracy & F1-Scores, and export `logistic_model.pkl` and `tfidf_vectorizer.pkl`.

## 🌐 Launching the Dashboard

Once trained, spin up the backend API and User Interface:
```bash
python server.py
```
Then navigate your browser to: **http://localhost:8000**

Paste any YouTube URL and watch your custom Logistic Regression model categorize the live sentiment and extract core topics dynamically!

## 💻 Core Stack

- **Machine Learning**: Scikit-Learn (Logistic Regression, SVM, TF-IDF Vectorization).
- **Backend Architecture**: FastAPI / Uvicorn.
- **Data Engineering**: Pandas, Google-API-Python-Client.
- **Frontend Design**: Vanilla JS & CSS (Glassmorphism UI).
