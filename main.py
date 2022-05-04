import flask
from PIL import Image
from flask import request
import os
import cv2
import cv2 as cv
import json
import random
from deeplab import DeeplabV3

app = flask.Flask(__name__)
deeplab = DeeplabV3()


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file_data']
    num = random.randint(1, 1000)
    dir = './tmp/'
    end = f.filename.split('.')[-1]
    img = os.path.join(dir, str(num) + end)
    f.save(img)
    image = Image.open(img)
    r_image = deeplab.detect_image(image)
    r_image.save('tmp.jpg')
    src = cv.imread('tmp.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 177, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max = 0
    final = None
    for i in contours:
        if len(i) >= max:
            final = i
            max = len(i)
    final = final.tolist()
    final = json.dumps(final)
    out = cv2.drawContours(src, contours, -1, (255, 0, 0), 3)
    cv2.imwrite('static/tmp.png', out)
    return final


@app.route('/')
def index():
    return flask.render_template('index.html')


def main():
    app.run()


if __name__ == '__main__':
    main()
