"""
Flask API serving the sports celebrity face recognition application.
Routes requests to the model for image classification.
Architecture note: Model loading logic is decoupled into `util.py` for cleaner API routing.
"""
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__, template_folder="templates")
util.load_saved_artifacts()


@app.route("/", methods=["GET", "POST"])
def index() -> Any:
    return render_template("index.html")


@app.route("/assets/<path:filename>")
def assets(filename: str) -> Any:
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    return send_from_directory(static_dir, filename)


@app.route("/classify_image", methods=["POST"])
def classify_image() -> Any:
    image_data = request.form["image_data"]
    response = jsonify(util.classify_image(image_data))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server for Sports Celebrity Face Recognition")
    app.run(port=5000)
