import flask
from flask import request
import os
import cv2
import json

app = flask.Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload():

    return '123'


@app.route('/')
def index():
    return 'Hello'

def main():
    app.run()


if __name__ == '__main__':
    main()
