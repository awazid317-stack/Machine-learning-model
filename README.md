# Machine-learning-model 

# 🏠 House Price Predictor

A Machine Learning project that predicts house prices using the **California Housing Dataset**. The project implements and compares two regression algorithms — **Linear Regression** and **Random Forest Regression** — to understand how different models perform on house-price prediction.

## 📌 Project Overview

The model takes various housing-related features such as:

* Median income
* House age
* Average number of rooms
* Average number of bedrooms
* Population
* Average household size
* Geographic location (latitude and longitude)

and predicts the **median house price** for a given area.

## 🤖 Machine Learning Models

Two regression models are trained and evaluated:

### 1. Linear Regression

A simple baseline regression model used to understand the linear relationship between housing features and price.

### 2. Random Forest Regressor

An ensemble-based machine learning algorithm that combines multiple decision trees to capture more complex, non-linear relationships in the dataset.

## 🔄 Project Workflow

```text
California Housing Dataset
          ↓
Data Loading & Preprocessing
          ↓
Feature / Target Separation
          ↓
Train-Test Split
          ↓
 ┌──────────────────────┐
 │  Linear Regression   │
 └──────────────────────┘
          ↓
     Predictions
          ↓
   Model Evaluation
          
 ┌──────────────────────┐
 │ Random Forest        │
 │ Regressor            │
 └──────────────────────┘
          ↓
     Predictions
          ↓
   Model Evaluation
          ↓
Actual vs Predicted Visualization
```

## 📊 Model Evaluation

The models are evaluated using:

* **Mean Absolute Error (MAE)** — measures the average absolute difference between actual and predicted prices.
* **R² Score** — measures how well the model explains the variation in house prices.

An **Actual vs Predicted** scatter plot is also generated to visually compare the model's predictions with the actual values.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Linear Regression
* Random Forest Regression

## 🎯 Objective

The main objective of this project is to build a basic end-to-end **regression machine learning pipeline**, understand house-price prediction, compare different regression algorithms, and evaluate their performance using standard regression metrics.

## 🚀 Future Improvements

* Add a user-friendly prediction interface using **Streamlit**
* Perform feature engineering
* Tune Random Forest hyperparameters
* Add cross-validation
* Compare additional regression algorithms
* Save and load the trained model for real-time predictions
