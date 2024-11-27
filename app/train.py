import os
from .model import create_cnn_model, prepare_data_generators

def train_model():
    # Direktori data training
    train_dir = os.path.join(os.path.dirname(__file__), '../data/train')
    validation_dir = os.path.join(os.path.dirname(__file__), '../data/validation')
    
    # Membuat generator data
    train_generator, validation_generator = prepare_data_generators(train_dir, validation_dir)
    
    # Membuat model
    model = create_cnn_model()
    
    # Melatih model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=20
    )
    
    # Menyimpan model
    model_path = os.path.join(os.path.dirname(__file__), '../models/aloe_vera_classifier.h5')
    model.save(model_path)
    
    return history