import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_aloe_image(image_path):
    # Memuat model yang sudah dilatih
    model_path = os.path.join(os.path.dirname(__file__), '../models/aloe_vera_classifier.h5')
    model = load_model(model_path)
    
    # Memproses gambar
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Melakukan prediksi
    predictions = model.predict(img_array)
    class_labels = ['Sehat', 'Sakit']
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    return predicted_class, confidence