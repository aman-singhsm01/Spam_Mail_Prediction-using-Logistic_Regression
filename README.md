# Spam Mail Prediction using Logistic Regression

## Summary
This project implements a machine learning model for predicting whether an email is spam or ham (not spam). It utilizes a logistic regression algorithm trained on a dataset containing labeled emails. The dataset is preprocessed and transformed into feature vectors using TF-IDF vectorization. The trained model achieves high accuracy in distinguishing between spam and ham emails.

## Getting Started
To run this project locally, follow these steps:

### Prerequisites
- Python
- Libraries: numpy, pandas, scikit-learn

### Installation
1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.

### Usage
1. Prepare your dataset or use the provided `mail_data.csv`.
2. Run the Jupyter Notebook or Python script containing the code.
3. Input a new email text to predict whether it's spam or ham.

## Description
The code consists of the following main parts:
- **Data Preprocessing**: The dataset is loaded, cleaned, and split into training and testing sets.
- **Feature Extraction**: TF-IDF vectorization is applied to convert text data into numerical feature vectors.
- **Model Training**: Logistic regression model is trained on the training data.
- **Model Evaluation**: Accuracy is calculated on both training and testing data.
- **Prediction**: New email text is input, transformed into feature vectors, and fed into the trained model for prediction.

## Results
- The model achieves an accuracy of approximately 96.7% on the training data and 96.6% on the testing data.
- Predictions can be made for new email texts, accurately classifying them as spam or ham.


