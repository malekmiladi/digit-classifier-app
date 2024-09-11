from ml_model.model import L_layer_model

from flask import Flask
from flask import request, jsonify
from flask import render_template

from utils import extract_dataurl_content, normalize_image

app = Flask(__name__)
model = L_layer_model()
model.load_params("params_3L_784x256x10.json")

# TODO: Add routes, logic and page for inputting images and their labels and saving
# them as training data
# TODO: Add route, logic and page for training the model
# TODO: Add route, logic and page for creating models with different layers and
# training them and comparing them
# TODO: modify the save_image decorator to be able to save data either as train
# or test, and change project folder structure to be better

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def predict_digit():
    data_url = request.json["image"]

    encoded_image = extract_dataurl_content(data_url)
    normalized_image = normalize_image(encoded_image) / 255.
    transformed_image = normalized_image.reshape((1, 28 * 28 * 1)).T
    predicted_digit = model.predict(transformed_image, None)

    return jsonify(digit=int(predicted_digit[0]))