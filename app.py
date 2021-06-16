import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = pd.DataFrame([int_features],columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    prediction=model.predict_proba(final_features)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    if output>str(0.5):
        return render_template('index.html',pred='congrats\nProbability of getting approved {}'.format(output))
    else:
        return render_template('index.html',pred='sorry\n Probability of getting approved {}'.format(output))


if __name__ == "__main__":
    app.run()