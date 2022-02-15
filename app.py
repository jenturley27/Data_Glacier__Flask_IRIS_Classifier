import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    SW = request.form.get('Sepal_Width') 
    SL = request.form.get('Sepal_Length') 
    PW = request.form.get('Petal_Width')
    PL = request.form.get('Petal_Length')


    #create an array with pred and actual or feed a vector to Z to get a result for one 
    data = {'sepal_width' : SW, 'sepal_length' : SL, 'petal_width' : PW, 'petal_length': PL}
    vec = pd.DataFrame(data, index=[0])  
    final_measures = vec
    

    result = model.predict(final_measures)
    response = "An iris with sepal width of " + SW + ", sepal length of "+ SL + ", petal width of " + PW + ", petal length of " + PL+ " is most likely of type "+ result[0]

    

    #it was essential to set prediction_text (from template file) equal to output
    return render_template('index.html',  prediction_text = response)

if __name__ == "__main__":
    app.run(debug=True)