from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def get_trained_model(self):
        self.model = tf.keras.models.load_model(
            self.config.trained_model_path
        )

    def convert_images_to_rgb(self):
        def verify_and_convert_images(directory, target_mode='RGB'):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        try:
                            with Image.open(file_path) as img:
                                if img.mode not in ["RGB", "RGBA", "L"]:
                                    # print(f"Invalid color mode: {img.mode} for image: {file_path}")
                                    # Convert to the target mode
                                    img = img.convert(target_mode)
                                    img.save(file_path)
                        except (IOError, SyntaxError) as e:
                            print(f"Invalid image found: {file_path} - {e}")
                            os.remove(file_path)  # Optionally remove the corrupted file

        # Verify and convert images in train and test directories
        verify_and_convert_images(self.config.training_data, target_mode='RGB')
        verify_and_convert_images(self.config.testing_data, target_mode='RGB')


    def read_data(self):
        img_height = 28
        img_width = 28
        batch_size = self.config.params_batch_size
        seed_train_validation = 1 # Must be same for train_ds and val_ds
        shuffle_value = True

        train_generator = tf.keras.utils.image_dataset_from_directory(
            directory = self.config.training_data,
            image_size = (img_height, img_width),
            seed = seed_train_validation,
            shuffle = shuffle_value,
            labels='inferred',
            label_mode='int',
            ) # set as training data

        validation_generator = tf.keras.utils.image_dataset_from_directory(
            directory = self.config.testing_data,
            image_size = (img_height, img_width),
            seed = seed_train_validation,
            shuffle = shuffle_value,
            labels='inferred',
            label_mode='int',
        ) # set as validation data


        # Normalize the datasets
        normalization_layer = tf.keras.layers.Rescaling(1./255)

        self.train_generator_norm = train_generator.map(lambda x, y: (normalization_layer(x), y))
        self.validation_generator_norm = validation_generator.map(lambda x, y: (normalization_layer(x), y))


        # Split the validation dataset into test and validation datasets
        train_batches = tf.data.experimental.cardinality(self.train_generator_norm)
        self.train_ds = self.train_generator_norm.take((3 * train_batches) // 4)
        self.val_ds = self.train_generator_norm.skip((3 * train_batches) // 4)

        plt.figure(figsize=(15, 15))
        for images, labels in self.train_ds.take(1):
            print(images.shape, labels.shape)
            for i in range(10):
                ax = plt.subplot(5, 5, i + 1)
                plt.imshow(images[i].numpy())
                plt.title(train_generator.class_names[labels[i]])
                plt.axis("off")
        plt.show()


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self, callback_list: list):
        # Check if a GPU is available and set the device accordingly
        if tf.test.gpu_device_name():
            print('GPU found')
            device = '/device:GPU:0'  # Use the first GPU
        else:
            print('No GPU found')
            device = '/device:CPU:0'

        with tf.device(device):

            history = self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=self.config.params_epochs)
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )

    def test(self):
        test_loss, test_accuracy = self.model.evaluate(self.validation_generator_norm)
        print(f"Test accuracy: {test_accuracy}")

    def predict(self):
        # Collect images and labels from the dataset
        images_list = []
        labels_list = []

        for images, labels in self.validation_generator_norm:
            images_list.append(images.numpy())
            labels_list.append(labels.numpy())

        # Convert lists to NumPy arrays
        X = np.concatenate(images_list, axis=0)
        y = np.concatenate(labels_list, axis=0)

        # Run predictions
        predictions = self.model.predict(X)

        # Convert predictions to class labels
        predicted_labels = np.argmax(predictions, axis=1)

        # Convert true labels from one-hot encoded vectors to class labels
        true_labels = y

        # Calculate accuracy
        accuracy = np.mean(predicted_labels == true_labels)

        print(f"Accuracy: {accuracy * 100:.2f}%")