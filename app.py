# app.py — Flask Web Interface for Fake News Detector
# Run: python app.py
# Open: http://localhost:5000

import os
from flask import Flask, render_template, request, jsonify
from predict import load_model, predict
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model ONCE at startup
print("Loading model... please wait")
model = load_model()
print("Model ready!\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    title   = request.form.get("title", "").strip()
    comment = request.form.get("comment", "").strip()

    if not title:
        return jsonify({"error": "Please enter a headline"}), 400

    # Handle image upload
    image_path = None
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

    result = predict(model, title, comment, image_path=image_path)

    # Clean up uploaded image after prediction
    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    return jsonify(result)


if __name__ == "__main__":
    print("Starting web server...")
    print("Open in browser : http://localhost:5000")
    print("Local network   : find your IP with 'ipconfig' and open http://YOUR-IP:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)