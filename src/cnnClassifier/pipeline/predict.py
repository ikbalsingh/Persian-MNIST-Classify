import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (28,28))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = test_image / 255.0 
        print(test_image.shape)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)
        # if result[0] == 0:
        #     prediction = '0'
        return [{ "image" : str(result[0])}]
        # elif result[0] == 1:
        #     prediction = '1'
        #     return [{ "image" : prediction}]
        # elif result[0] == 2:
        #     prediction = '2'
        #     return [{ "image" : prediction}]
        # elif result[0] == 3:
        #     prediction = '3'
        #     return [{ "image" : prediction}]
        # else:
        #     prediction = 'Tulip'
        #     return [{ "image" : prediction}]
