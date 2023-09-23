import os
import cv2
import numpy as np
import base64
import jwt

import service as service

from flask import Flask, request, jsonify

from configuration.serviceConfiguration import ServiceConfiguration
from configuration.keyConfiguration import KeyConfiguration

from logger.logger import Logger, LogFileHandler

app = Flask(__name__)

PYTHON_HOST, PYTHON_PORT = ServiceConfiguration.load_config()
PY_SECRET, US_SECRET, MS_SECRET = KeyConfiguration.load_config()

imageClassifier = service.ImageClassifier()


@app.route('/classificator/classify', methods=['POST'])
def classify():
    try:
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

        # saveImage(img, 'C')

        # Take white text and center it 

        # Get image center
        h, w = img.shape
        center = (w // 2, h // 2)

        # Get image center of mass
        M = cv2.moments(img)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Get offset
        dX = center[0] - cX
        dY = center[1] - cY

        # Shift image
        M = np.float32([[1, 0, dX], [0, 1, dY]])
        img = cv2.warpAffine(img, M, (w, h))

        #saveImage(img, 'Y') # 
        # I S O V Q Y Z
        if trainMode == 'true':
            result = imageClassifier.train(img, label)
            return jsonify({'status': 'ok', 'result': str(result)}), 200

        result = imageClassifier.classify(img)

        return jsonify({'status': 'ok', 'result': str(result)}), 200
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/py/service/check', methods=['GET'])
def check():
    return jsonify({'status': 'ok'}), 200

@app.route('/structure/get', methods=['GET'])
def getStructure():
    return {'status': 'ok', 'result': imageClassifier.getStructure()}

@app.route('/logs/get', methods=['GET'])
def getLogs():
    data = request.get_json()
    token = data['token']
    if not authenticate(token):
        return {'status': 'error', 'message': 'Invalid token'}

    logs = LogFileHandler.get_instance().get_logs()


    if token == None:
        return {'status': 'error', 'message': 'No token provided'}

    return {'status': 'ok', 'result': logs}

@app.route('/logs/u/get', methods=['POST'])
def getLogsSingle():
    json_data = request.get_json()

    file_name = json_data['name']
    filter = json_data['filter']

    if file_name == None:
        return {'status': 'error', 'message': 'No file name provided'}
    
    log_data = LogFileHandler.get_instance().get_log(file_name)

    return jsonify({'status': 'ok', 'result': log_data}), 200


def decodeBase64(b64):
    data = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def authenticate(token):
    try:
        jwt.decode(token, PY_SECRET, algorithms=['HS256'])
        return True
    except:
        return False
    
def saveImage(image, name):
    # Check if image with that name exists and add number to it
    i = 0
    while True:
        if os.path.isfile('./images/'+name + str(i) + '.png'):
            i += 1
        else:
            break
    
    print ("Saving image: ", name + str(i) + '.png')
    cv2.imwrite('./images/'+ name + str(i) + '.png', image)
        

if __name__ ==  '__main__':
    
    logger = Logger.get_instance()
    logger.info("Server started", origin="AIService", action="Init", port=PYTHON_PORT, host=PYTHON_HOST)
    app.run(host=PYTHON_HOST, port=PYTHON_PORT, debug=True, use_reloader=False)


    