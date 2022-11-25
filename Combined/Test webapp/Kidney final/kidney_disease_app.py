import numpy as np
import pickle
from flask import Flask, request, render_template


model = pickle.load(open(
    'D:/NSU COURSES/NSU SEMESTER 14/CSE445/Test webapp/Kidney final/kidney.pkl', 'rb'))

app = Flask(__name__)

# home


@app.route('/')
def home():
    return render_template('kidney.html')

# predict


@app.route('/predictKidney', methods=['POST'])
def predict():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    print(array_features)
    # Predict features
    prediction = model.predict(array_features)
    print(prediction)

    #output = prediction
    # Output
    if prediction == 1:
        return render_template('kidney.html', result='The patient is not likely to have Kidney disease!')
    else:
        return render_template('kidney.html', result='The patient is 98 percent likely to have Kidney disease!')


if __name__ == '__main__':
    app.run()
