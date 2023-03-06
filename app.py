from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('model1.h5', compile=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        result = 'Pneumonia'
    else:
        result = 'Normal'
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
