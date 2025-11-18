
import io
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model_bundle = joblib.load("savedmodel.pth")
model = model_bundle["model"]

# Convert uploaded image to the format expected by the model."""
def preprocess_image(file_storage):    
    image_bytes = file_storage.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L") 
    img = img.resize((64, 64))  
    img_arr = np.array(img, dtype=np.float32) / 255.0    
    return img_arr.reshape(1, -1)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            error = "Please choose an image file."
        else:
            try:
                features = preprocess_image(request.files["image"])
                pred_class = int(model.predict(features)[0])
                prediction = f"Predicted class: {pred_class}"
            except Exception as e:
                error = f"Error during prediction: {e}"

    return render_template("upload.html", prediction=prediction, error=error)


if __name__ == "__main__":    
    app.run(host="0.0.0.0", port=5000, debug=True)