from flask import Flask,render_template, request
import numpy as np
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('malaria.h5')

@app.route('/')
def home():
    return render_template("home.html")

def classprediction(process_file):
    result = model.predict(process_file)
    result = result > 0.5 
    return result

def data_process(image):
    image = image.reshape(1,128,128,3)
    image = image/255
    result = classprediction(image)
    return result

@app.route('/result',methods=['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        file = request.files['image_input']
        img = load_img(file,target_size=(128,128))
        img = img_to_array(img)
        result = data_process(img)
        if result == True:
            prediction = "Congratulation, You are uninfected"
        else:
            prediction = "Your result is 'Parasitized'"
        print (prediction)
        return render_template('result.html',prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)
    


    
    
