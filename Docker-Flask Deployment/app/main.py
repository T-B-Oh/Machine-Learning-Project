from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64

app = Flask(__name__)

model = tf.keras.models.load_model(os.path.join("model", "model_5.h5"))

cifar10_labels = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="Please upload the file")

    file = request.files["file"]
    image_bytes = file.read()
    image_type = file.mimetype or "image/png"
    image_url = f"data:{image_type};base64," + base64.b64encode(image_bytes).decode("utf-8")
    input_data = preprocess_image(image_bytes)

    prediction = model.predict(input_data)
    predicted_class = int(np.argmax(prediction[0]))
    predicted_label = cifar10_labels[predicted_class]

    return render_template(
        "index.html",
        prediction=predicted_label,
        image_url=image_url,
        file_name=file.filename
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
