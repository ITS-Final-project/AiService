import cv2
import numpy as np
import base64
from PIL import Image

import service as service

from flask import Flask, request, jsonify

app = Flask(__name__)

PYTHON_PORT = 3002
PYTHON_HOST = 'localhost'

imageClassifier = service.ImageClassifier()

@app.route('/classificator/classify', methods=['POST'])
def classify():
    json_data = request.get_json()

    url = json_data['data']['imgBase64']
    trainMode = json_data['data']['trainMode']
    label = json_data['data']['label']

    print("Train mode: ", trainMode)

    b64 = url.split(',')[1]
    
    img = decodeBase64(b64)

    # Shape image to 28x28
    img = cv2.resize(img, (28, 28))

    # Make np array
    img = np.array(img)

    # Make grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('preview.png', img)    

    if trainMode == 'true':
        result = imageClassifier.train(img, label)
        return jsonify({'status': 'ok', 'result': str(result)}), 200
    

    result = imageClassifier.classify(img)

    return jsonify({'status': 'ok', 'result': str(result)}), 200

@app.route('/py/service/check', methods=['GET'])
def check():
    return jsonify({'status': 'ok'}), 200

@app.route('/structure/get', methods=['GET'])
def getStructure():
    return {'status': 'ok', 'result': imageClassifier.getStructure()}

def decodeBase64(b64):
    data = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

if __name__ ==  '__main__':
    app.run(host=PYTHON_HOST, port=PYTHON_PORT, debug=True)

    