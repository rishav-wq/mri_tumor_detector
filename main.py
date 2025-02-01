from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Load the model
model = load_model("models/model.h5")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']


# Define the upload folder
UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Flask upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function for tumor prediction
def predict_tumor(image_path):
    image_size = 128
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(image_size, image_size))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        # Determine the class
        if class_labels[predicted_class_index] == 'notumor':
            return "No Tumor", confidence_score
        else:
            return f"Tumor: {class_labels[predicted_class_index]}", confidence_score
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Route for serving uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main route for handling uploads and predictions
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']

        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict results
            result, confidence = predict_tumor(file_location)

            # Return results along with image path for display
            return render_template('index.html', result=result, confidence=f'{confidence * 100:.2f}', file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10002)
