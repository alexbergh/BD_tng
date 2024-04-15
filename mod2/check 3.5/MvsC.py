import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# Функция для загрузки и аугментации данных
def prepare_data(train_dir, test_dir, img_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, test_generator

# Улучшенная полносвязная нейронная сеть
def improved_fcc_nn(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Улучшенная сверточная нейронная сеть
def improved_conv_nn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Функция для обучения модели
def train_model(model, train_data, test_data, epochs=50):
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs
    )
    return history

# Функция для визуализации результатов обучения
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Визуализация парсептрона
def plot_model_pars(model):
    plot_model(model, to_file='model_pars.png', show_shapes=True, show_layer_names=True)

# Подготовка данных
train_data, test_data = prepare_data('./train', './test')

# Обучение улучшенной полносвязной модели
improved_fcc_model = improved_fcc_nn(input_shape=(150, 150, 3))
improved_fcc_history = train_model(improved_fcc_model, train_data, test_data, epochs=50)
print("Обучение улучшенной полносвязной модели завершено.")
plot_history(improved_fcc_history)
plot_model_pars(improved_fcc_model)

# Обучение улучшенной сверточной модели
improved_conv_model = improved_conv_nn(input_shape=(150, 150, 3))
improved_conv_history = train_model(improved_conv_model, train_data, test_data, epochs=50)
print("Обучение улучшенной сверточной модели завершено.")
plot_history(improved_conv_history)
plot_model_pars(improved_conv_model)

# Вывод
# После обучения обеих моделей,```
