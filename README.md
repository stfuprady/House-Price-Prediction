# House Price Prediction

This project uses **Linear Regression** to predict house prices based on various features such as lot size, year built, and other important factors. The model was trained and evaluated on a dataset containing information about house sales.

## Overview
The goal of this project is to predict the sale price of a house given different features. The features used include various property attributes, such as the size of the lot, the year the house was built, the number of bedrooms, and more.

## Key Results
- **Mean Squared Error**: 1,418,361,538
- **Mean Absolute Error**: 23,474
- **R² Score**: 0.81, indicating the model explains 81% of the variance in house prices.

## Installation and Requirements

To run this project locally, you'll need to install the following dependencies:
- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install them using `pip`:
```bash
pip install -r requirements.txt
```

## How to Run the Project
Clone the repository:

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```
Navigate to the project directory:
```bash
cd House-Price-Prediction
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the main script to predict house prices:
```bash
python house_price_prediction.py
```
## Project Workflow
#### Data Preprocessing:
-Missing values were handled using imputation (filling with the median).
-Only numerical features were used after dropping categorical ones.
-Feature scaling was applied to standardize the data.
Model:
Linear Regression was used for training the model.
Model performance was evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.
Results:
The model achieved an R² score of 0.81, meaning 81% of the variance in house prices is explained by the model.
Future Improvements
Advanced Models: Experimenting with more complex models like Random Forest or Gradient Boosting could improve the model's accuracy.
Feature Engineering: Additional feature engineering techniques could be applied to increase prediction power.
