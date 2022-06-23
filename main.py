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


@app.route('/v2/upload', methods=['POST'])
def uploadV2():
    f = request.files['file_data']
    num = random.randint(1, 1000)
    dir = './tmp/'
    end = f.filename.split('.')[-1]
    img = os.path.join(dir, str(num) + end)
    f.save(img)
    image = Image.open(img)
    r_image, pr3 = deeplab.detect_image(image)
    r_image.save('tmp.jpg')
    src = cv.imread('tmp.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 177, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _max = 0
    final = {}
    _index = -1
    tmp = None
    for id, i in enumerate(contours):
        if len(i) >= _max:
            tmp = i.squeeze()
            _max = len(i)
            _index = id
    if tmp is None:
        final['contours'] = None
        return final
    final['contours'] = tmp.tolist()
    ans = 0
    for i in tmp:
        ans += pr3[i[1]-1][i[0]-1]
    final['credibility'] = ans.item() / tmp.shape[0]
    final = json.dumps(final)
    out = cv2.drawContours(src, contours, _index, (255, 0, 0), 3)
    cv2.imwrite('static/tmp.png', out)
    return final


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file_data']
    num = random.randint(1, 1000)
    dir = './tmp/'
    end = f.filename.split('.')[-1]
    img = os.path.join(dir, str(num) + end)
    f.save(img)
    image = Image.open(img)
    r_image, pr3 = deeplab.detect_image(image)
    r_image.save('tmp.jpg')
    src = cv.imread('tmp.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 177, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _max = 0
    final = None
    _index = -1
    for id, i in enumerate(contours):
        if len(i) >= _max:
            final = i
            _max = len(i)
            _index = id
    final = final.tolist()
    final = json.dumps(final)
    out = cv2.drawContours(src, contours, _index, (255, 0, 0), 3)
    cv2.imwrite('static/tmp.png', out)
    return final


@app.route('/')
def index():
    return flask.render_template('index.html')


def main():
    app.run()


if __name__ == '__main__':
    main()
