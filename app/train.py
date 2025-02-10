import os
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-interaktif
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
from .model import create_cnn_model, prepare_data_generators
import tensorflow as tf

def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1).numpy())

def train_model():
    # Direktori data training
    train_dir = os.path.join(os.path.dirname(__file__), '../data/train')
    validation_dir = os.path.join(os.path.dirname(__file__), '../data/validation')
    
    # Membuat generator data
    train_generator, validation_generator = prepare_data_generators(train_dir, validation_dir)
    
    # Membuat model
    model = create_cnn_model()
    
    # Melatih model
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size


    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=20,
        callbacks=[LearningRateScheduler(lr_scheduler)]
    )
    
    # Menyimpan model dalam format .keras
    model_path = os.path.join(os.path.dirname(__file__), '../models/aloe_vera_classifier.h5')
    model.save(model_path)

    # Evaluasi model dan buat laporan
    cm_path = os.path.join(os.path.dirname(__file__), '../static/reports/confusion_matrix.png')
    cr_path = os.path.join(os.path.dirname(__file__), '../static/reports/classification_report.png')
    create_evaluation_report(model, validation_generator, cm_path, cr_path)
    
    # Buat grafik pelatihan
    plot_path = os.path.join(os.path.dirname(__file__), '../static/reports/training_plot.png')
    plot_training_history(history, plot_path)

    # Buat laporan pelatihan
    report_path = os.path.join(os.path.dirname(__file__), '../static/reports/training_report.png')
    create_training_report(history, report_path)

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
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def create_training_report(history, save_path):
    """Buat dan simpan laporan pelatihan dan validasi dalam bentuk tabel tanpa judul dan margin."""
    # Ekstrak data dari history
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    # Data untuk Training Report
    training_data = {
        'Epoch': epochs_range,
        'Training Loss': history.history['loss'],
        'Training Accuracy': history.history['accuracy'],
        'Training Precision': history.history.get('precision', [None] * len(epochs_range)),
        'Training Recall': history.history.get('recall', [None] * len(epochs_range))
    }
    df_training = pd.DataFrame(training_data)

    # Data untuk Validation Report
    validation_data = {
        'Epoch': epochs_range,
        'Validation Loss': history.history['val_loss'],
        'Validation Accuracy': history.history['val_accuracy'],
        'Validation Precision': history.history.get('val_precision', [None] * len(epochs_range)),
        'Validation Recall': history.history.get('val_recall', [None] * len(epochs_range))
    }
    df_validation = pd.DataFrame(validation_data)

    # Simpan Training Report sebagai gambar terpisah
    training_report_path = save_path.replace(".png", "_training.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')  # Matikan axis
    table = ax.table(cellText=df_training.values, colLabels=df_training.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_training.columns))))
    
    # Hilangkan margin dan padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(training_report_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Simpan Validation Report sebagai gambar terpisah
    validation_report_path = save_path.replace(".png", "_validation.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')  # Matikan axis
    table = ax.table(cellText=df_validation.values, colLabels=df_validation.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_validation.columns))))
    
    # Hilangkan margin dan padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(validation_report_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# def create_evaluation_report(model, validation_generator, cm_path, cr_path):
#     """Buat dan simpan Classification Report sebagai gambar terpisah."""
#     # Prediksi pada data validasi
#     val_steps = validation_generator.samples // validation_generator.batch_size
#     y_true = validation_generator.classes[:val_steps * validation_generator.batch_size]
#     y_pred = model.predict(validation_generator, steps=val_steps)
#     y_pred_classes = np.argmax(y_pred, axis=1)
    
#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred_classes)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validation_generator.class_indices.keys())
    
#     # Plot Confusion Matrix
#     disp.plot(cmap=plt.cm.Blues, values_format='d')
#     plt.title('Confusion Matrix')
#     plt.savefig(cm_path)
#     plt.close()

#     # Classification Report
#     report = classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys(), output_dict=True)

#     # Konversi laporan ke dalam gambar menggunakan Matplotlib
#     plot_classification_report(report, cr_path)