#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import Callback

base_dir = "./"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Data augmentation and data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(112, 112),
    batch_size=10,
    class_mode='binary' # Using binary classification
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(112, 112),
    batch_size=10,
    class_mode='binary' # Using binary classification
)

# Network based on VGG19
conv_base = VGG19(include_top=False, input_shape=(112, 112, 3), weights='imagenet')
for layer in conv_base.layers:
    layer.trainable = False

z = conv_base.output
z = GlobalAveragePooling2D()(z)
z = Dense(128, activation='relu')(z)
z = Dropout(0.5)(z)
predictions = Dense(1, activation='sigmoid')(z)

model = Model(inputs=conv_base.input, outputs=predictions)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Logging to text file
log_filename = "training_log.txt"

class LogCallback(Callback):
    def __init__(self, log_filename):
        super().__init__()
        self.log_filename = log_filename
        self.log_file = open(self.log_filename, 'w')
    
    def on_epoch_end(self, epoch, logs=None):
        log_message = f"Epoch {epoch + 1}, Loss: {logs['loss']}, Accuracy: {logs['accuracy']}, Val Loss: {logs['val_loss']}, Val Accuracy: {logs['val_accuracy']}\n"
        print(log_message, end='')
        self.log_file.write(log_message)

    def on_train_end(self, logs=None):
        self.log_file.close()

    def print_final_results(self, history):
        # Assuming history is a dictionary containing 'accuracy', 'val_accuracy', 'loss', 'val_loss'
        final_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        print(f"\nFinal Training Accuracy: {final_acc}, Final Validation Accuracy: {final_val_acc}")
        print(f"Final Training Loss: {final_loss}, Final Validation Loss: {final_val_loss}\n")

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[LogCallback(log_filename)], # Ensure LogCallback uses the file properly
    verbose=1
)

log_callback = LogCallback(log_filename)
log_callback.print_final_results(history.history)

model.save("vgg19_chihuahua_vs_muffin.keras")

# Getting data from training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Saving graphs to PDF
with PdfPages('training_results.pdf') as pdf:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].plot(epochs, acc, 'b', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].legend()

    ax[1].plot(epochs, loss, 'b', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation Loss')
    ax[1].set_title('Training and Validation Loss')
    ax[1].legend()
    
    pdf.savefig(fig)
    plt.close(fig)
       
    plt.show()
    
    pdf.savefig(fig)
    plt.close(fig)
