import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model

model = load_model("model.h5")

app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "Face With Mask"
	elif classNo==1:
		return "Face Without Mask"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((128, 128))
    image=np.array(image)
    # np.reshape(image, [1,128,128,3])
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    result01=np.argmax(result,axis=1)
    return result01


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)