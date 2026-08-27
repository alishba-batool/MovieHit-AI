# 🎬 MovieHit AI — Movie Success Predictor

MovieHit AI is an end-to-end machine learning project that predicts whether a movie is likely to become a commercial hit based on its production and release characteristics.

The project uses historical movie data to train and compare multiple machine learning classification models. The best-performing model is then deployed as an interactive Streamlit web application.

---

## 🚀 Live Demo

👉 **Try MovieHit AI:**  
https://moviehit-ai-4vbwkpur9zcb5sjgdmwxcv.streamlit.app/

---

## 📌 Project Overview

Movie production involves significant financial investment, but predicting whether a movie will become commercially successful is difficult.

MovieHit AI explores whether machine learning can provide an early prediction of movie success using information that can be available before release.

The application allows users to enter movie details such as:

- Production budget
- Runtime
- Release year
- Release month
- Main genre
- Number of production companies
- Number of production countries
- Number of cast members
- Number of crew members

The trained model then predicts whether the movie is likely to be:

- 🔥 **A Hit**
- 📉 **Not a Hit**

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze historical movie data.
2. Clean and preprocess the dataset.
3. Perform exploratory data analysis.
4. Engineer useful movie-related features.
5. Define a movie success classification target.
6. Train multiple machine learning models.
7. Compare model performance.
8. Select the best-performing model.
9. Save the trained model.
10. Build an interactive Streamlit application.
11. Deploy the application online.

---

## 📊 Dataset

The project uses the **TMDB 5000 Movie Dataset**.

The dataset contains information about movies including:

- Budget
- Revenue
- Genres
- Release dates
- Production companies
- Production countries
- Cast
- Crew
- Popularity
- Ratings

For this project, the model focuses mainly on features that can be available before a movie's release.

---

## 🔄 Machine Learning Workflow

The project follows a complete machine learning pipeline:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Target Creation
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Saving
   ↓
Streamlit Deployment
