import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam    
import pathlib

dataset_dir = pathlib.Path('dataset5').with_suffix('')
image_count = len(list(dataset_dir.glob("*/*.png")))
print(f"Всего изображений: {image_count}")

# Определяет количество образцов, которые будут 
# распространяться через сеть одновременно (4, 8, 16, ..)
batch_size = 16

#Размеры 1 изображения входных данных
img_width = 640
img_height = 640

# Создание обучающего набора данных
train_ds = tf.keras.utils.image_dataset_from_directory(
	dataset_dir,
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

# Создание валидационного набора данных
val_ds = tf.keras.utils.image_dataset_from_directory(
	dataset_dir,
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

# Получение и вывод на экран названий классов
class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Кэширование, перемешивание и предварительная обработка данных
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
model = Sequential([
    
	layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

	# аугментация
	layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
	layers.experimental.preprocessing.RandomRotation(0.1),
	layers.experimental.preprocessing.RandomZoom(0.1),
	layers.experimental.preprocessing.RandomContrast(0.2),

	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Conv2D(128, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Conv2D(256, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	# регуляризация
    # Для предотвращения переобучения
	layers.Dropout(0.2),

	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(num_classes)
])

# Задайте скорость обучения
learning_rate = 0.00001

# Создайте оптимизатор с заданной скоростью обучения
optimizer = Adam(learning_rate=learning_rate)

model.compile(
	optimizer = optimizer,
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])

# количество эпох тренировки
epochs = 100
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

