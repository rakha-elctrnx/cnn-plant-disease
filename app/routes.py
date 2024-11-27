import os
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
from .train import train_model
from .predict import predict_aloe_image

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/train', methods=['GET', 'POST'])
def train_data():
    if request.method == 'POST':
        # Memulai proses pelatihan
        history = train_model()
        return render_template('train_data.html', training_complete=True)
    return render_template('train_data.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Cek apakah file ada
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # Cek apakah nama file kosong
        if file.filename == '':
            return redirect(request.url)
        
        # Jika file valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Prediksi gambar
            prediction, confidence = predict_aloe_image(filepath)
            
            return render_template('result.html', 
                                   filename=filename, 
                                   prediction=prediction, 
                                   confidence=confidence)
    
    return render_template('upload.html')