import numpy as np
import pickle
from flask import Flask, request, render_template


modelKidney = pickle.load(open(
    'D:/NSU COURSES/NSU SEMESTER 14/CSE445/Test webapp/Main/kidney.pkl', 'rb'))
modelHeart = pickle.load(open(
    'D:/NSU COURSES/NSU SEMESTER 14/CSE445/Test webapp/Main/eda-pickle.pkl', 'rb'))


app = Flask(__name__)

# home


@app.route('/home')
def dashboard():
    
    dekhi = render_template(
        'D:/NSU COURSES/NSU SEMESTER 14/CSE445/Test webapp/Main/home.html')
    return dekhi

# predict


@app.route('/predictKidney', methods=['POST'])
# def dashboard():
#     dekhi = render_template(
#         'D:/NSU COURSES/NSU SEMESTER 14/CSE445/Test webapp/Main/home.html')
#     return dekhi
# def diseaseDetection():
#     predictKidney()
#     predictHeart()
def predictKidney():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # print(array_features)
    # Predict features
    prediction = modelKidney.predict(array_features)
   # print(prediction)

    #output = prediction
    # Output
    if prediction == 1:
        return render_template('kidney.html', result='The patient is not likely to have Kidney disease!')
    else:
        return render_template('kidney.html', result='The patient is 98 percent likely to have Kidney disease!')


@app.route('/predictHeart', methods=['POST'])
def predictHeart():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # print(array_features)
    # Predict features
    prediction = modelHeart.predict(array_features)
    # print(prediction)

    #output = prediction

    # Output
    if prediction == 1:
        return render_template('Heart Disease Classifier.html',
                               result='The patient is 98 percent likely to have heart disease!')
    else:
        return render_template('Heart Disease Classifier.html',
                               result='The patient is not likely to have heart disease!')


if __name__ == '__main__':
    app.run()
