from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import torch
from torchvision import transforms
from PIL import Image




app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

model1 = torch.load('model1.pth', map_location=torch.device('cpu'))
model1.eval()  # Set the model to evaluation mode


app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'wav'}  # Only allow .wav files

# Check if the uploaded file is of allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    # Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        # Create a new user
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return render_template('register.html')

    return render_template('register.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['username'] = username
            flash('Login successful!', 'success')
            return render_template('dashboard1.html')
        else:
            flash('Invalid username or password!', 'danger')

    return render_template('register.html')

def audi_model():
    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_data = train_datagen.flow_from_directory(
        'lung_sound_data/sound_images',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    total_sample=train_data.n
    batch_size = 32
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout added to prevent overfitting
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, steps_per_epoch=int(total_sample/batch_size), epochs=5, verbose=0)
    return model
            

def wav_to_spectrograms(wav_file, save_path):
    y, sr = librosa.load(wav_file, sr=None)  # Ensure consistent sampling rate
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize spectrogram
    mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))

    # Save as image
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def predict_audio(wav_file, model):
    # Convert WAV to spectrogram
    spectrogram_path = 'static/lung_res.jpg'
    wav_to_spectrograms(wav_file, spectrogram_path)
    
    # Load the spectrogram image
    img = load_img(spectrogram_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]        
    return class_index, confidence
# Dashboard

@app.route('/audio')
def audio1():
    return render_template('audio.html')

@app.route('/image')
def image1():
    return render_template('image.html')


@app.route('/dashboard')
def dashboard():    
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    flash('Please log in to access the dashboard.', 'warning')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        audio = request.files['audio_file']
        if audio and allowed_file(audio.filename):
            # Save the uploaded file to the specified folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
            audio.save(filename)
        model = audi_model()
        class_index, confidence = predict_audio(filename, model)
        class_labels = ['Abnormal', 'Normal']
        prediction_label = class_labels[class_index]
        if prediction_label == "Abnormal":
            suggestion = "Your respiratory condition seems abnormal. We recommend taking an X-ray for further examination and to get a more accurate diagnosis."
        else:
            suggestion = "Your respiratory system seems healthy. Keep maintaining a good lifestyle for continued well-being."

        return render_template('audio.html', username="User", prediction=prediction_label, suggestion=suggestion)
        

    return render_template('dashboard.html', username=session['username'])

    
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  # Normalization mean (adjust if trained differently)
        std=[0.5, 0.5, 0.5]    # Normalization std (adjust if trained differently)
    ),
])


@app.route('/image', methods=['GET','POST'])
def image():
    if request.method == 'POST':
        xray_image = request.files['xray_image']
        #if xray_image and allowed_xray_file(xray_image.filename):
            # Save X-ray image
        xray_filename = os.path.join(app.config['UPLOAD_FOLDER'], xray_image.filename)
        xray_image.save(xray_filename)
            
            # Process the X-ray image and make prediction
        image = Image.open(xray_filename).convert("RGB")  # Ensure the image is RGB
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Perform inference
        with torch.no_grad():  # Disable gradient computation for inference
                outputs = model1(input_tensor)
                _, predicted_class = torch.max(outputs, 1)  # Get the index of the highest score

            # Print the predicted class
            #print(f"Predicted class index: {predicted_class.item()}")

            # Example class names (replace with your actual class names)
        class_names = ["Corona Virus Disease","Normal","Tuberculosis","Viral Pneumonia"]
        xray_prediction = class_names[predicted_class.item()]
            # Process X-ray prediction
        if xray_prediction == 'Normal':  # Assuming 1 represents lung disease detected
                suggestion = "The X-ray appears normal. Continue maintaining good health."   
             
        else:
                suggestion = "The X-ray indicates a possible lung disease. Please consult a doctor immediately for further tests."
                
        return render_template('image.html', username="User", prediction=xray_prediction, suggestion=suggestion)
            

# Logout
@app.route('/dashboard1')
def dashboard1():
    return render_template('dashboard1.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database and tables
    app.run(debug=False)
