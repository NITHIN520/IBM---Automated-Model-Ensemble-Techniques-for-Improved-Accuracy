from flask import Flask, render_template, request
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

app = Flask(__name__)

# Load dataset (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Define individual models
model1 = LogisticRegression(random_state=42)
model2 = DecisionTreeClassifier(random_state=42)
model3 = SVC(probability=True, random_state=42)

# Create an ensemble model using VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('lr', model1),
    ('dt', model2),
    ('svc', model3)
], voting='soft')

# Train the ensemble model
ensemble_model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get values from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare the input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict using the trained ensemble model
        prediction = ensemble_model.predict(input_data)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
