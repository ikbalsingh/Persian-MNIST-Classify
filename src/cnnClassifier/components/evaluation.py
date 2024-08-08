import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):
        img_height = 28
        img_width = 28
        batch_size = self.config.params_batch_size
        seed_train_validation = 1 # Must be same for train_ds and val_ds
        shuffle_value = True

        self.validation_generator = tf.keras.utils.image_dataset_from_directory(
            directory = self.config.testing_data,
            image_size = (img_height, img_width),
            seed = seed_train_validation,
            shuffle = shuffle_value,
            labels='inferred',
            label_mode='int',
        ) # set as validation data

        # Normalize the datasets
        normalization_layer = tf.keras.layers.Rescaling(1./255)

        self.validation_generator_norm = self.validation_generator.map(lambda x, y: (normalization_layer(x), y))


    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.validation_generator_norm)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def predict_and_plot(self):
        num_images = 10 
        # Load the trained model
        # Read and preprocess images
        def load_and_preprocess_image(img_path, target_size=(28, 28)):
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = img_array/255.0  # Apply preprocessing specific to your model
            return img_array

        # Collect image paths and labels
        image_paths = []
        true_labels = []
        class_names = sorted(os.listdir(self.config.training_data))  # Sorted to maintain consistent order

        for class_index, class_name in enumerate(class_names):
            class_folder = os.path.join(self.config.training_data, class_name)
            if os.path.isdir(class_folder):
                for filename in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, filename)
                    image_paths.append(img_path)
                    true_labels.append(class_index)

        # Create an index array and shuffle it
        indices = np.arange(len(image_paths))
        # np.random.seed()  # For reproducibility
        np.random.shuffle(indices)

        # Select 10 images and corresponding labels using shuffled indices
        selected_indices = indices[:num_images]
        selected_image_paths = [image_paths[i] for i in selected_indices]
        selected_true_labels = [true_labels[i] for i in selected_indices]

        # Load and preprocess selected images
        images = np.array([load_and_preprocess_image(img_path) for img_path in selected_image_paths])

        # Run predictions
        predictions = self.model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)

        # Plot images with predictions and actual labels
        plt.figure(figsize=(15, 10))
        for i in range(num_images):
            plt.subplot(2, 5, i + 1)
            img_display = images[i] * 255  # Reverse normalization
            img_display = np.clip(img_display, 0, 255)  # Ensure pixel values are in [0, 255]
            plt.imshow(img_display.astype(np.uint8))  # Convert to uint8 for display
            plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[selected_true_labels[i]]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    