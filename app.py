from flask import Flask, render_template, request, Response, redirect, url_for
import numpy as np
import keras.models
import cv2
import time
from keras.utils import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt

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

NAME = ""
def isVerified(file, name):
    for line in open(file, 'r'):
        if(line[:len(line) - 1] == name):
            return "verified"
    return "unverified"


# Generating txt file with emotion labels
# Return: name of the text file
def emotionDetection(Videofilename):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('model.h5')

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    predictions = []
    cap = cv2.VideoCapture(Videofilename)
    textfile = None
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if(ret == True):

        # FRAME_COUNT += 1
            time.sleep(1 / fps)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # print(label)
                    predictions.append(label)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # cv2.imshow('Emotion Detector', frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                # print(predictions)

                break
        else:
            textfile = open("suhali_file.txt", "w")
            for element in predictions:
                textfile.write(element + "\n")
            textfile.close()
            break

    return textfile.name

# Creating Pie chart from o/p of emotiondetection function.
def createPi(textfilename):
    dict_emotions = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}

    # for emotion in dict_emotions:

    for line in open(textfilename, 'r'):
        dict_emotions[line[:len(line) - 1]] = dict_emotions[line[:len(line) - 1]] + 1

    print(dict_emotions)

    labels = []
    sizes = []

    max = -1
    label_max = "some"

    for x, y in dict_emotions.items():
        if (y > max):
            max = y
            label_max = x
        if (y != 0):
            labels.append(x)
            sizes.append(y)

    # print(label_max)
    # Plot
    plt.pie(sizes, labels=labels)
    plt.axis('equal')
    plt.savefig("static/emotionAnalysis.jpg")
    return label_max


def detect_face():
    global NAME
    while True:
        success, frame = camera.read()
        if (not success):
            break
        else:
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                if(name=="Unknown"):
                    return "unverified"
                else:
                    NAME = name
                    return "verified"
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        break


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
                if(name=='Unknown'):
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                else:
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
                # time = time+1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


## Restful APIs

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

@app.route('/programs')
def programs():
    return render_template('programs.html')

@app.route('/cells')
def cells():
    return render_template('cells.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/stream')
def stream():
    return render_template('stream.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cell9')
def cell9():
    status = isVerified("cell9.txt", NAME)
    if(status == "verified"):
        text_filename = emotionDetection("Test_video.mp4")
        label = createPi(text_filename)
        return render_template('cell9.html', emotion=label)
    else:
        return redirect(url_for('cells'))

@app.route("/getimage")
def get_img():
    return "emotionAnalysis.jpg"

@app.route('/requests',methods=['POST','GET'])
def tasks():
    if request.method == 'POST' or request.method == 'GET':

        if request.form.get('Verify') == 'Verify':

            x=detect_face()
            if (x == "unverified"):
                print("unverified here")
                return render_template('homepage.html')
            elif (x == "verified"):
                print("here")
                with app.app_context():
                    return redirect(url_for('cells'))
    print("here")
    return render_template('stream.html')

if __name__ == "__main__":
    app.run(debug=True)
