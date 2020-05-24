from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)

loaded_model = pickle.load(open('titanicmodeldeploy.pkl','rb'))

@app.route('/')
def home():
    123
    return render_template('titanic.html')
def predictor(to_predict_list):
    predictedresult = loaded_model.predict(np.array(to_predict_list).reshape(1,len(to_predict_list)))
    return predictedresult[0]

@app.route('/result', methods = ['POST'])

def result():
    prediction=''
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = [float(i) for i in list(to_predict_list.values())]
        result = predictor(to_predict_list)

        if int(result)==0:
            prediction = 'not survived'
        elif int(result)==1:
            prediction = 'survived'
        return render_template('result.html',prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    