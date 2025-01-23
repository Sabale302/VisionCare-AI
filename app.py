import os
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from werkzeug.utils import secure_filename


# Load API configuration
API_KEY = "DR7ojKd15aBOiYF6ba37UDSwBVcmvQqtq9BdkyCxnHMa"

# Get the authentication token
token_response = requests.post(
    'https://iam.cloud.ibm.com/identity/token',
    data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
)
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

FlaskApp = Flask(__name__)
CORS(FlaskApp)  # Allow cross-origin requests

# Load class labels dynamically
class_labels = {
    0: "Normal",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Cataract"
}

@FlaskApp.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is part of the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file to a temporary location
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)

        # Open and preprocess the image
        image = Image.open(filepath).resize((256, 256))
        image_array = np.array(image)

        if image_array.shape[-1] != 3:
            return jsonify({"error": "Image must have 3 channels (RGB)"}), 400

        # Prepare input data
        input_data = np.expand_dims(image_array, axis=0)
        payload_scoring = {"input_data": [{"values": input_data.tolist()}]}

        # Add the version parameter to the model API request URL
        model_url = 'https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/visioncare_ai_v1/predictions?version=2021-05-01'

        # Make the API request to the ML model
        response_scoring = requests.post(
            model_url,
            json=payload_scoring,
            headers=header
        )

        # Log the full response for debugging
        print("Model response:", response_scoring.json())

        # Ensure predictions exist in the response
        response_json = response_scoring.json()
        predictions = response_json.get('predictions', [])
        if not predictions or not predictions[0].get('values'):
            raise ValueError("Invalid response: No predictions found")

        # Access the first sublist in values
        values = predictions[0]['values'][0]

        # Extract probabilities and predicted index
        prediction_values = values[0]  # Probabilities (list)
        predicted_index = values[1]  # Predicted index (int)

        # Map the predicted index to a class label
        predicted_label = class_labels.get(predicted_index, "Unknown")

        # Log and return the prediction
        print("Prediction values:", prediction_values)
        print("Predicted index:", predicted_index)
        print("Predicted label:", predicted_label)

        return jsonify({"predicted_class": predicted_label, "probabilities": prediction_values})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    FlaskApp.run(debug=True)
