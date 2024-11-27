import os
from .model import create_cnn_model, prepare_data_generators
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

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

    # Evaluasi model dan buat laporan
    cm_path = os.path.join(os.path.dirname(__file__), '../static/reports/confusion_matrix.png')
    cr_path = os.path.join(os.path.dirname(__file__), '../static/reports/classification_report.png')
    create_evaluation_report(model, validation_generator, cm_path, cr_path)
    
    # Buat grafik pelatihan
    plot_path = os.path.join(os.path.dirname(__file__), '../static/reports/training_plot.png')
    plot_training_history(history, plot_path)

    return history

def create_evaluation_report(model, validation_generator, cm_path, cr_path):
    """Buat dan simpan Confusion Matrix serta Classification Report sebagai gambar terpisah."""
    # Prediksi pada data validasi
    val_steps = validation_generator.samples // validation_generator.batch_size
    y_true = validation_generator.classes[:val_steps * validation_generator.batch_size]
    y_pred = model.predict(validation_generator, steps=val_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validation_generator.class_indices.keys())
    
    # Plot Confusion Matrix
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(cm_path)
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys(), output_dict=True)

    # Konversi laporan ke dalam gambar menggunakan Matplotlib
    plot_classification_report(report, cr_path)

def plot_classification_report(report, save_path):
    """Buat gambar Classification Report."""
    # Konversi dictionary ke dalam format tabel
    df = pd.DataFrame(report).transpose()

    # Plot menggunakan Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Simpan sebagai gambar
    plt.title("Classification Report")
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_path):
    """Membuat dan menyimpan grafik pelatihan."""
    # Ekstrak data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs_range = range(len(loss))

    # Plot grafik
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    # Simpan grafik ke file
    plt.savefig(save_path)
    plt.close()