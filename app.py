from flask import Flask, render_template, request, Response, redirect
import numpy as np
import keras.models
import cv2
import time

model = keras.models.load_model('model.h5')
app = Flask(__name__)

from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("faces/")

# Load Camera
cap = cv2.VideoCapture(0)
pred = "Unknown"


camera = cv2.VideoCapture(0)

def generate_frames():

    status = "unverified"
    time = 0
    while True:

        ## read the camera frame
        success, frame = camera.read()
        if (not success):
            break
        else:
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                # if(name != "Unknown"):
                #     status = "verified"
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                # time = time+1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # if(status == "verified"):
    #     print(status)
    #    return redirect('/analysis')
    # else:
    #     print(status)
    #     return redirect('/analysis')

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/cells')
def cells():
        return render_template('cells.html')

@app.route('/analysis')
def analysis():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

