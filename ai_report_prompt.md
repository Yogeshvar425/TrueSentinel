# Instructions for AI Assistant

You are tasked with writing a comprehensive academic project report for a professor based on the following project context. Please generate a detailed, structured, and professional report suitable for a university-level submission.

## Project Details
**Project Title:** TrueSentiment: Full-Stack Machine Learning Pipeline
**Project Objective:** A full-stack application that acts as a complete pipeline from Data Harvesting to Model Training and Web Deployment for YouTube sentiment analysis.

## Core Features
1. **From-Scratch ML Pipeline:** Custom trained Machine Learning models (Logistic Regression & SVM) instead of pre-trained APIs.
2. **Automated Data Harvesting:** YouTube Data API v3 integration to fetch live, real-world comments across diverse videos.
3. **Smart Lexicon Labeling:** Robust NLP lexical ruleset (including catching double-negatives like "not bad") to establish ground-truth labels automatically.
4. **Model Training & Comparison:** Text vectorization using TF-IDF (supporting Bigrams), strictly training Linear Regression and Support Vector Machines with Class Weight Balancing to handle skewed datasets.
5. **Live Interactive Dashboard:** FastAPI backend with an aesthetically pleasing, Glassmorphism-styled frontend UI that feeds user inputs through custom `.pkl` models for real-time analysis.

## Tech Stack
- **Machine Learning:** Scikit-Learn (Logistic Regression, SVM, TF-IDF Vectorization)
- **Backend:** FastAPI, Uvicorn, Python
- **Data Engineering:** Pandas, Google-API-Python-Client
- **Frontend:** Vanilla JavaScript, HTML5, CSS3 (Modern Glassmorphism aesthetics)

## Architecture Flow
1. **`train.py` (Data & Training Pipeline)**: Harvests YouTube data locally, generates `custom_training_data.csv`, trains the models, reports Accuracy & F1-Scores, and exports `logistic_model.pkl` and `tfidf_vectorizer.pkl`.
2. **`server.py` (Backend API)**: A FastAPI server that loads the trained `.pkl` models and exposes endpoints for inference.
3. **Frontend (`static/`)**: An intuitive dashboard where users can paste a YouTube URL to get dynamic live sentiment analysis and core topic extraction.

## Required Report Structure
Please ensure the generated report includes the following sections:
1. **Abstract:** High-level summary of the problem, methodology, and outcome.
2. **Introduction:** Context, motivation, and objectives of the TrueSentiment project.
3. **System Architecture:** Detailed explanation of the full-stack architecture (Frontend, Backend, Data Engineering, ML Pipeline).
4. **Methodology:** 
   - Data Collection Strategy (YouTube API)
   - Data Preprocessing & Lexical Labeling rules
   - Feature Engineering (TF-IDF, Bigrams)
   - Model Selection and Training (Logistic Regression & SVM, Class Balancing)
5. **Implementation Details:** Key technologies utilized and their integration.
6. **Results & Evaluation:** Discussion of training metrics (Accuracy, F1-Scores) and real-time dashboard performance.
7. **Conclusion & Future Work:** Summary of achievements and potential enhancements.

## Additional Instructions
- The tone must be formal, academic, and highly technical.
- Expand heavily upon the machine learning concepts utilized (e.g., explain why TF-IDF and SVM/Logistic Regression are suitable for text classification, and the importance of Class Balancing).
- Emphasize the "built-from-scratch" nature of the pipeline as a key differentiator rather than relying on black-box APIs.
- Feel free to infer standard web-development and ML boilerplate explanations to pad the report appropriately for academic submission.
