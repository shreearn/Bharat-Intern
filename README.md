# Bharat-Intern
# 1. Stock Price Prediction Using LSTM

Introduction 

This code uses Long Short-Term Memory (LSTM) neural networks to predict stock prices. LSTM networks are particularly useful for time series prediction due to their ability to maintain long-term dependencies.


Dataset

The dataset used in this project is NSE-TATAGLOBAL stock dataset, which contains historical stock prices of Tata Global Beverages from the National Stock Exchange of India.


Requirements

- Python 3
- Pandas
- Matplotlib
- NumPy
- Scikit-learn
- TensorFlow


Model Architecture: The LSTM model consists of three LSTM layers with 50 units each, followed by a dense layer with 1 unit. It is trained using the Adam optimizer and mean squared error loss function.

Results: The model achieves a root mean squared error (RMSE) of X on the training data and Y on the test data. The predictions are visualized alongside the actual stock prices.



# 2. Titanic Survival Prediction

Introduction 

This code aims to predict the survival of passengers aboard the Titanic using machine learning models. The dataset contains various features such as socio-economic status, age, gender, and more, which are used to predict whether a passenger survived or not.

Dataset

The dataset used in this project is the famous Titanic dataset, which includes information about passengers such as their name, age, gender, socio-economic status (SES), ticket class, cabin number, port of embarkation, and whether they survived or not.

Requirements

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost

Data Wrangling and Visualization

1. The data is first loaded into a pandas DataFrame.
2. Missing values are visualized using the missingno library.
3. The distribution of features and survival rates are visualized using seaborn and matplotlib.


Machine Learning Models
Several machine learning models are used for prediction, including:

- Logistic Regression
- Perceptron
- Stochastic Gradient Descent (SGD) Classifier
- Support Vector Machine (SVM)
- Random Forest Classifier
- K-Nearest Neighbors (KNN) Classifier
- Gaussian Naive Bayes Classifier
- Decision Tree Classifier
- CatBoost Classifier


Model Evaluation: The models are evaluated using cross-validation to ensure robustness and avoid overfitting.

Hyperparameter Tuning: GridSearchCV is used for hyperparameter tuning to find the best parameters for each model.

Results: The performance of each model is evaluated based on metrics such as accuracy, precision, recall, and F1-score. The most important factors contributing to survival are identified through feature importance analysis.
