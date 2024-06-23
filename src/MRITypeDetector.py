import string
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt


class MRITypeDetector:
    def __init__(self, modelName: string) -> None:
        print("Num GPUs Available: ", len(
            tf.config.list_physical_devices('GPU')))
        # Set the seed value for experiment reproducibility.
        seed = 1842
        tf.random.set_seed(seed)
        np.random.seed(seed)
        # Turn off warnings for cleaner looking notebook
        warnings.simplefilter('ignore')
        self.modelName = modelName
        self.model = None

    # 'data_cleaned/Train'    'data_cleaned/Train'
    def train(self, dirTrain: string, dirValidate: string) -> None:
        # define image dataset
        # Data Augmentation
        self.image_generator = ImageDataGenerator(
            rescale=1/255,
            rotation_range=10,  # rotation
            width_shift_range=0.2,  # horizontal shift
            height_shift_range=0.2,  # vertical shift
            zoom_range=0.2,  # zoom
            horizontal_flip=True,  # horizontal flip
            brightness_range=[0.2, 1.2],  # brightness
            validation_split=0.2,)

        # Train & Validation Split
        self.train_dataset = self.image_generator.flow_from_directory(batch_size=32,
                                                                      directory=dirTrain,
                                                                      shuffle=True,
                                                                      target_size=(
                                                                          224, 224),
                                                                      subset="training",
                                                                      class_mode='categorical')

        self.validation_dataset = self.image_generator.flow_from_directory(batch_size=32,
                                                                           directory=dirValidate,
                                                                           shuffle=True,
                                                                           target_size=(
                                                                               224, 224),
                                                                           subset="validation",
                                                                           class_mode='categorical')

        self.model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=[224, 224, 3]),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (2, 2), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (2, 2), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=3,
                                                 restore_best_weights=True)

        self.model.fit(self.train_dataset, epochs=20,
                       validation_data=self.validation_dataset, callbacks=callback)

        loss, accuracy = self.model.evaluate(self.validation_dataset)
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

        self.model.save(self.modelName)

    # 'data_cleaned/scraped_images'
    def classify(self, dirClass: string, filesubFolder: string):
        # Organize data for our predictions
        image_generator_submission = ImageDataGenerator(rescale=1/255)
        submission = image_generator_submission.flow_from_directory(
            directory=dirClass,
            shuffle=False,
            target_size=(224, 224),
            class_mode=None)

        if self.model is None:
            self.model = keras.models.load_model(self.modelName)

        onlyfiles = [f.split('.')[0] for f in os.listdir(os.path.join(dirClass+filesubFolder))
                     if os.path.isfile(os.path.join(os.path.join(dirClass+filesubFolder), f))]
        submission_df = pd.DataFrame(onlyfiles, columns=['images'])
        submission_df[['la_eterna', 'other_flower']
                      ] = self.model.predict(submission)
        submission_df.head()

        submission_df.to_csv('submission_file.csv', index=False)


classifier = MRITypeDetector('klasyfikatorTypu')
# classifier.train('data_cleaned/Train', 'data_cleaned/Train')
classifier.classify('data_cleaned/scraped_images', '/image_files')
