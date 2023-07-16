import base64
import cv2
import io

import numpy as np

from PIL import Image

class StreamProcessor:
    # singleton
    @classmethod
    def processStream(cls, data):
        img64 = base64.b64decode(data)

        #ok
        image = Image.open(io.BytesIO(img64))
        image_np = np.array(image)

        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        flipped_image = cv2.flip(image_gray, 1)

        resized_image = cv2.resize(flipped_image, (80, 60)).T

        # cut image by half of height
        cut_image = resized_image[round(60 / 2):, :]

        _, mask = cv2.threshold(cut_image, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
        cut_image_masked = cv2.bitwise_and(cut_image, mask)

        return cut_image_masked
