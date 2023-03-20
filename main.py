import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
model_path="C:\\Users\\aleja\\OneDrive\\Escritorio\\Tensorflow\\Comida\\model"
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


BATCH_SIZE=100
IMG_SHAPE=150
img_train_dir="C:\\Users\\aleja\\OneDrive\\Escritorio\\Tensorflow\\Comida\\all_imgs_train"
 # img_val_dir  ="C:\\Users\\aleja\\OneDrive\\Escritorio\\TensorFlow\\Comida\\all_imgs_val"
train_dir='C:\\Users\\aleja\\OneDrive\\Escritorio\\Tensorflow\\Comida\\train.csv'
test_dir='C:\\Users\\aleja\\OneDrive\\Escritorio\\Tensorflow\\Comida\\test.csv'
traindf =pd.read_csv(train_dir, usecols = [1,2], dtype=str)
testdf =pd.read_csv(test_dir,dtype=str)

def remove_ext(fn):
    return fn.split("/", 1)[1]


traindf["path_img"]= traindf["path_img"].apply(remove_ext)

image_gen_train = ImageDataGenerator(rescale=1./255, zoom_range=0.5, horizontal_flip=True, rotation_range=45, validation_split=0.2, vertical_flip=True
                                 , shear_range=0.2, width_shift_range=0.2,height_shift_range=0.2, test_split=0.1)

train_generator=image_gen_train.flow_from_dataframe(dataframe=traindf, directory=img_train_dir, x_col='path_img', y_col="label", shuffle=True,
                                                    batch_size=BATCH_SIZE, target_size=(IMG_SHAPE, IMG_SHAPE),class_mode='sparse', subset='training')
val_generator=image_gen_train.flow_from_dataframe(dataframe=traindf, directory=img_train_dir, x_col='path_img', y_col="label",
                                                 batch_size=BATCH_SIZE, target_size=(IMG_SHAPE, IMG_SHAPE),class_mode='sparse', subset='validation')

image_gen_train.flow

augmented_images = [train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)

model= keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

    tf.keras.layers.Conv2D(64, 3,padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128,3,padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(514, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(8, activation='softmax')
]
)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.01, patience=10)
callback2 = tf.keras.callbacks.ModelCheckpoint(filepath="C:\\Users\\aleja\\OneDrive\\Escritorio\\Tensorflow\\Comida\\saves",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

epochs=120
model.fit(train_generator,steps_per_epoch=int(np.ceil(train_generator.n / float(BATCH_SIZE))),epochs=epochs, validation_data=val_generator,
                            validation_steps=int(np.ceil(val_generator.n / float(BATCH_SIZE))), callbacks=[callback1, callback2])


model.save(model_path)

estdf["path_img"]= testdf["path_img"].apply(remove_ext)
print(testdf)

# Carga el dataframe con los nombres de las imágenes

# Crea el generador de imágenes
datagen = ImageDataGenerator(rescale=1./255)

# Carga las imágenes desde el directorio y prepara los datos sin etiquetas
test_generator = datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=img_dir,
        x_col="path_img",
        y_col=None,
        target_size=(150, 150),
        batch_size=100,
        class_mode=None, label_mode=None)

# Carga el modelo entrenado
model = tf.keras.models.load_model(model_path)

# Hace predicciones en los datos de prueba
predictions =  tf.argmax(model.predict(test_generator),1)

# Imprime las predicciones
print(predictions)

import json
testdf2=pd.read_csv(test_dir,dtype=str, usecols=[0])
# Crea un diccionario de Python con los nombres de las imágenes y las predicciones

predictions_dict = dict(zip(testdf2["idx_test"], predictions.numpy().tolist()))
# Guarda el diccionario en un archivo JSON
with open("C:\\Users\\aleja\\OneDrive\\Escritorio\\Tensorflow\\Comida\\Predicciones\\pred.json.txt", 'w') as f:
    json.dump(predictions_dict, f)