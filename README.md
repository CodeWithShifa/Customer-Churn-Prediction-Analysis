# 📉 Customer Churn Prediction (Deep Learning)

Losing customers is expensive. This project uses **Artificial Neural Networks (ANN)** to predict the likelihood of a customer leaving a service (Churn) based on their usage patterns, age, and account information.

## 🧠 Model Architecture
This project uses a Multi-Layer Perceptron (Neural Network) with:
- **Input Layer:** Takes features like Credit Score, Tenure, Balance, etc.
- **Hidden Layers:** Uses 'ReLU' activation to find complex patterns in customer behavior.
- **Output Layer:** Uses a 'Sigmoid' function to give a probability (0 to 1) of churn.



## 🛠️ Key Features
* **Data Encoding:** Converting categorical data (like Gender or Country) into numbers using One-Hot Encoding.
* **Feature Scaling:** Using `StandardScaler` to ensure all data is on the same scale for the Neural Network.
* **Performance Metrics:** Using a Confusion Matrix to see how many customers were correctly identified as "at risk".

## 📁 Repository Content
* `churn_deep_learning.py`: The main script with the Neural Network logic.
* `Churn_Modelling.csv`: The dataset containing customer records.

## 💻 Tech Stack
* **Python**
* **TensorFlow / Keras**
- **Pandas & NumPy**
- **Scikit-Learn**# Customer-Churn-Prediction-Analysis
A Deep Learning (ANN) model to predict customer attrition and identify at-risk customers using behavioral data
