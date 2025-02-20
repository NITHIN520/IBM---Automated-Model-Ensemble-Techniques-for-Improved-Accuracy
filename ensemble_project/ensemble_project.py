# ensemble_project.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_and_save_model():
    # Load dataset (Iris dataset for example)
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define individual models
    model1 = LogisticRegression(random_state=42)
    model2 = DecisionTreeClassifier(random_state=42)
    model3 = SVC(probability=True, random_state=42)

    # Create an ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(estimators=[
        ('lr', model1),
        ('dt', model2),
        ('svc', model3)
    ], voting='soft')  # 'soft' for probability-based voting

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Save the model (you can also save it as a pickle or joblib file for future use)
    return ensemble_model

def predict(model, features):
    # Predict on new data (features should be a list or array)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]
