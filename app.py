from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Function to preprocess input and one-hot encode categorical variables
def preprocess_input(age, sex, bmi, children, smoker, region):
    # Encode sex: 1 for male, 0 for female
    sex = 1 if sex == 'male' else 0

    # Encode smoker: 1 for yes, 0 for no
    smoker = 1 if smoker == 'yes' else 0

    # One-hot encode region
    region_northeast = 1 if region == 'northeast' else 0
    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0

    # Return the input in the correct format (9 features)
    return np.array([[age, bmi, children, sex, smoker, region_northeast, region_northwest, region_southeast, region_southwest]])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    age = float(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    # Process the input data
    input_data = preprocess_input(age, sex, bmi, children, smoker, region)

    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Return the prediction as a JSON response
    return jsonify(results=prediction)

if __name__ == '__main__':
    app.run(debug=True)
