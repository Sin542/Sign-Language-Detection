from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
import io
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

app.static_folder = 'static'

UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
MODEL_FOLDER = os.path.join(app.static_folder, 'models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# ===================== DATABASE MODELS =====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # Add this line

class VideoLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    filename = db.Column(db.String(120))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class TranslationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    translated_text = db.Column(db.String(120))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AnimationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    sign_text = db.Column(db.String(120))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class LoginHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# ===================== MODEL LOADING =====================
model = None
class_names = None
scaler = None

def load_all_assets():
    global model, class_names, scaler
    try:
        model_path = os.path.join(MODEL_FOLDER, 'sign_model.h5')
        labels_path = os.path.join(MODEL_FOLDER, 'labels.npy')
        scaler_path = os.path.join(MODEL_FOLDER, 'scaler.pkl')

        model = load_model(model_path)
        class_names = np.load(labels_path)
        scaler = joblib.load(scaler_path)

        print("Model and assets loaded successfully.")
    except Exception as e:
        print("Model load error:", e)

load_all_assets()

# ===================== MEDIAPIPE HANDS SETUP =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ===================== LOGIN REQUIRED CHECK =====================
@app.before_request
def require_login():
    allowed_routes = ['login', 'register', 'static', 'test_db']
    if request.endpoint not in allowed_routes and 'username' not in session:
        return redirect(url_for('login'))

# ===================== ROUTES =====================

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    user = User.query.filter_by(username=session.get('username')).first()
    if request.method == 'POST':
        if user:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            # Validate inputs here if needed
            if username:
                user.username = username
                session['username'] = username
            if email:
                user.email = email
                session['email'] = email
            if password:
                user.password = password

            db.session.commit()
            flash("Profile updated successfully.")
            return redirect(url_for('home'))

    return render_template('home.html', username=session.get('username'), user=user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not username or not password or not confirm_password:
            flash("Please fill all fields.")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for('register'))

        if User.query.filter((User.email == email) | (User.username == username)).first():
            flash("Email or Username already exists.")
            return redirect(url_for('register'))

        new_user = User(email=email, username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please login.")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash("Please enter email and password.")
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['username'] = user.username
            session['email'] = user.email
            db.session.add(LoginHistory(username=user.username))
            db.session.commit()
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials.")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/detection')
def detection():
    return render_template('detection.html', username=session.get('username'))

@app.route('/animation')
def animation():
    return render_template('animation.html', username=session.get('username'))

@app.route('/get_animation_files')
def get_animation_files():
    folder = os.path.join(app.static_folder, 'avatars')
    files = os.listdir(folder) if os.path.exists(folder) else []
    gifs = [f for f in files if f.endswith('.gif')]
    glbs = [f for f in files if f.endswith('.glb')]
    return jsonify({'gifs': gifs, 'glbs': glbs})

@app.route('/dashboard')
def dashboard():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))

    videos = VideoLog.query.filter_by(username=username).order_by(VideoLog.timestamp.desc()).all()
    translations = TranslationLog.query.filter_by(username=username).order_by(TranslationLog.timestamp.desc()).all()
    animations = AnimationLog.query.filter_by(username=username).order_by(AnimationLog.timestamp.desc()).all()
    logins = LoginHistory.query.filter_by(username=username).order_by(LoginHistory.timestamp.desc()).all()

    return render_template('dashboard.html',
        video_pairs=[(v, v.timestamp.strftime('%Y-%m-%d %H:%M:%S')) for v in videos],
        translation_pairs=[(t, t.timestamp.strftime('%Y-%m-%d %H:%M:%S')) for t in translations],
        animation_pairs=[(a, a.timestamp.strftime('%Y-%m-%d %H:%M:%S')) for a in animations],
        login_pairs=[(l, l.timestamp.strftime('%Y-%m-%d %H:%M:%S')) for l in logins],
        username=username
    )

@app.route('/admin')
def admin_dashboard():
    user = User.query.filter_by(username=session.get('username')).first()
    if not user or not user.is_admin:
        flash("Access denied: Admins only.")
        return redirect(url_for('home'))

    users = User.query.all()
    videos = VideoLog.query.order_by(VideoLog.timestamp.desc()).all()
    translations = TranslationLog.query.order_by(TranslationLog.timestamp.desc()).all()
    animations = AnimationLog.query.order_by(AnimationLog.timestamp.desc()).all()
    logins = LoginHistory.query.order_by(LoginHistory.timestamp.desc()).all()

    return render_template('admin_dashboard.html',
        users=users,
        video_logs=videos,
        translation_logs=translations,
        animation_logs=animations,
        login_history=logins
    )

@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    username = request.form['username']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_new_password']

    if new_password != confirm_password:
        flash("Passwords do not match.")
        return redirect(url_for('login'))

    user = User.query.filter_by(username=username).first()
    if user:
        user.password = generate_password_hash(new_password)
        db.session.commit()
        flash("Password updated successfully.")
    else:
        flash("Username not found.")
    return redirect(url_for('login'))


# ===================== UPLOAD VIDEO (single route) =====================
@app.route('/upload_video', methods=['POST'])
def upload_video():
    print("Upload video endpoint called")
    file = request.files.get('video')
    username = session.get('username')
    print(f"File: {file}, Username: {username}")

    if file and username:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        ext = os.path.splitext(original_filename)[1] or '.webm'

        filename = f"{username}_{timestamp}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Saving video to: {filepath}")

        try:
            file.save(filepath)
            db.session.add(VideoLog(username=username, filename=filename))
            db.session.commit()
            print("Video saved and logged successfully")
            return jsonify({'status': 'success', 'filename': filename}), 200
        except Exception as e:
            print(f"Error saving video: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to save video: {e}'}), 500
    else:
        print("No file or user not logged in")
        return jsonify({'status': 'error', 'message': 'No file uploaded or user not logged in'}), 400


@app.route('/update_profile', methods=['POST'])
def update_profile():
    user = User.query.filter_by(username=session.get('username')).first()
    if user:
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if username:
            user.username = username
            session['username'] = username
        if email:
            user.email = email
            session['email'] = email
        if password:
            user.password = password

        db.session.commit()
        flash("Profile updated successfully.")
    else:
        flash("User not found.")

    return redirect(url_for('home'))


# ===================== PREDICTION ROUTE =====================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'prediction': 'No image data'}), 400

        # Decode base64 image data
        header, encoded = data.split(',', 1)
        img_bytes = base64.b64decode(encoded)

        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.flip(img_bgr, 1)  # horizontal flip to mirror

        # Process with MediaPipe
        results = hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return jsonify({'prediction': 'No Hand Detected'})

        landmark_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            single_hand = []
            for lm in hand_landmarks.landmark:
                single_hand.extend([lm.x, lm.y, lm.z])
            landmark_data.append(single_hand)

        while len(landmark_data) < 2:
            landmark_data.append([0.0] * 63)  # pad second hand landmarks with zeros

        input_data = np.array(landmark_data[0] + landmark_data[1]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        prediction_prob = model.predict(input_scaled)[0]
        predicted_label = class_names[np.argmax(prediction_prob)]

        # Log the translation
        db.session.add(TranslationLog(username=session.get('username', 'unknown'), translated_text=predicted_label))
        db.session.commit()

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'prediction': 'Error', 'error': str(e)}), 500

# ===================== LOG TRANSLATION =====================
@app.route('/log_translation', methods=['POST'])
def log_translation():
    text = request.json.get('text')
    if text and 'username' in session:
        db.session.add(TranslationLog(username=session['username'], translated_text=text))
        db.session.commit()
        return jsonify({'status': 'logged'})
    return jsonify({'status': 'error'}), 400

# ===================== LOG ANIMATION =====================
@app.route('/log_animation', methods=['POST'])
def log_animation():
    sign = request.json.get('sign')
    if sign and 'username' in session:
        db.session.add(AnimationLog(username=session['username'], sign_text=sign))
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

# ===================== UPLOAD MODEL =====================
@app.route('/upload_model', methods=['POST'])
def upload_model():
    file = request.files.get('model')
    if file and file.filename.endswith('.h5'):
        filepath = os.path.join(MODEL_FOLDER, 'sign_model.h5')
        file.save(filepath)
        load_all_assets()
        return jsonify({'status': 'Model uploaded and reloaded successfully'})
    return jsonify({'status': 'Invalid file'}), 400

# ===================== TEST DATABASE CONNECTION =====================
@app.route('/test_db')
def test_db():
    try:
        db.session.execute('SELECT 1')
        return '✅ Database connected successfully!'
    except Exception as e:
        return f'❌ Database connection failed: {e}'

if __name__ == '__main__':
    app.run(debug=True)
