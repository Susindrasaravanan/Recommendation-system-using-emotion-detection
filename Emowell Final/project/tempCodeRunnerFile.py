from flask import Flask, render_template, request, redirect, session, url_for, Response, jsonify, flash
from flask_pymongo import PyMongo
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure MongoDB connection
app.config['MONGO_URI'] = 'mongodb://localhost:27017/emowell'  # Replace with your MongoDB URI
mongo = PyMongo(app)

# Load models and data
face_classifier = cv2.CascadeClassifier(r'Emowell\final project emotion\project\haarcascade_frontalface_default.xml')
classifier = load_model(r'Emowell\final project emotion\project\model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load data from CSV files
def load_csv_data():
    try:
        movies_df = pd.read_csv(r'Emowell\final project emotion\project\movies.csv')
        music_df = pd.read_csv(r'Emowell\final project emotion\project\music.csv')
        games_df = pd.read_csv(r'Emowell\final project emotion\project\games.csv')
        novels_df = pd.read_csv(r'Emowell\final project emotion\project\novels.csv')
        return movies_df, music_df, games_df, novels_df
    except Exception as e:
        logging.error(f"Error loading CSV files: {str(e)}")
        return None, None, None, None

movies_df, music_df, games_df, novels_df = load_csv_data()

current_recommendations = {}

def get_recommendations(emotion):
    recommendations = {
        'movies': movies_df[movies_df['Emotion'] == emotion].to_dict(orient='records'),
        'music': music_df[music_df['Emotion'] == emotion].to_dict(orient='records'),
        'games': games_df[games_df['Emotion'] == emotion].to_dict(orient='records'),
        'novels': novels_df[novels_df['Emotion'] == emotion].to_dict(orient='records')
    }
    logging.debug(f"Recommendations for {emotion}: {recommendations}")
    return recommendations

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = mongo.db.users
        login_user = users.find_one({'username': request.form['username']})

        if login_user:
            # Here you might want to add password hashing for security
            if request.form['password'] == login_user['password']:
                session['username'] = request.form['username']
                return redirect(url_for('index'))
            else:
                flash('Incorrect password. Please try again.', 'error')
        else:
            flash('Username not found. Please try again or register.', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            # Here you might want to add password hashing for security
            users.insert_one({'username': request.form['username'], 'password': request.form['password']})
            session['username'] = request.form['username']
            return redirect(url_for('index'))
        else:
            flash('That username already exists!', 'error')

    return render_template('signup.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    logging.debug('Logging out user')
    session.pop('username', None)
    return redirect(url_for('login'))

video_capture_flag = True

def generate_frames():
    global current_recommendations, video_capture_flag
    cap = cv2.VideoCapture(0)
    while video_capture_flag:
        success, frame = cap.read()
        if not success:
            break
        else:
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
                    current_recommendations = get_recommendations(label)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()  # Release the camera


@app.route('/video_feed')
def video_feed():
    if 'username' in session:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return redirect(url_for('login'))
    
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global video_capture_flag
    video_capture_flag = True  # Reset the flag to start the detection process again
    return redirect(url_for('index'))  # Redirect to the index page


@app.route('/get_recommendations')
def get_recommendations_route():
    global video_capture_flag
    recommendations = current_recommendations
    logging.debug(f"Current Recommendations: {recommendations}")
    video_capture_flag = False  # Stop the video capture
    return jsonify(recommendations)



if __name__ == "__main__":
    app.run(debug=True)
