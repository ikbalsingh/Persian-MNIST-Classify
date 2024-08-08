import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = Sequential()

        # Add convolutional layer
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add another convolutional layer
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the output and feed it into a dense layer
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))

        # Add the output layer
        self.model.add(Dense(self.config.params_classes, activation='softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])     

        self.save_model(path=self.config.base_model_path, model=self.model)
    
    def update_base_model(self):
        self.full_model = self.model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    


