import os
import secrets

import werkzeug

import functions
import cv2

from flask import Flask, flash, request, redirect, url_for, send_from_directory, abort
from flask_cors import CORS

import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("./assets/img2cp-firebase-adminsdk-hiwt0-273dac6ac0.json")
firebase_admin.initialize_app(cred)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
CORS(app)


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_auth_request(e):
    return 'Authentication Failed', 500


@app.route('/uploadFiles', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    try:
        decoded_token = firebase_admin.auth.verify_id_token(request.headers["idtoken"])
        uid = decoded_token['uid']
        request.user = uid
    except Exception as e:
        print(e)
        abort(500)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = request.headers["idtoken"] + "_" + secrets.token_hex(16) + "." + file.filename.split('.')[-1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return {"success": 1, "file": filename}


@app.route('/identifyPoints', methods=['GET'])
def identify_points():
    filename = secrets.token_hex(16) + ".png"
    cImage = cv2.imwrite("uploads/" + filename, functions.identifyPoints(fileName=request.args["img"],
                                                                         pointQuality=float(request.args["pQuality"]),
                                                                         minDistance=int(request.args["minDist"])))
    return {"success": 1, "file": filename}
