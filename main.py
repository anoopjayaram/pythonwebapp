import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    _, buffer = cv2.imencode('.png', processed_image)
    processed_image_data = base64.b64encode(buffer).decode('utf-8')
    processed_image_data = f"data:image/png;base64,{processed_image_data}"
    return jsonify({'processed_image': processed_image_data})

if __name__ == '__main__':
    app.run(debug=True)
