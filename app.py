from flask import Flask, request, render_template
import keras
import numpy as np
from keras.applications.mobilenet import preprocess_input
from PIL import Image
import tensorflow as tf
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("final_clean_model.h5")

class_names = ['cats', 'dogs']


def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)

    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction

    # 🔥 STRONGER threshold
    if confidence < 0.90:
        return None

    if prediction < 0.5:
        return f"🐱 Cat ({confidence:.2f})"
    else:
        return f"🐶 Dog ({confidence:.2f})"


import base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file.stream)

        # 🔥 Convert image to base64
        import io
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        result = predict_image(img)

        if result is None:
            return render_template('index.html', error=True, image=img_str)

        return render_template('index.html', prediction=result, error=False, image=img_str)

    return render_template('index.html', prediction=None, error=False)


if __name__ == "__main__":
    app.run(debug=True)