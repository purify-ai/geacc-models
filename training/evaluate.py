#    Copyright 2019 Purify Foundation
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Evaluate GACC trained model

from keras import layers, models, callbacks, backend
from keras.optimizers import Adam
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

### Hyperparameters
output_classes = 2
learning_rate = 4e-5  #0.00004
img_size = 224 # img width and height
batch_size = 128
epochs = 90
resume_model = False
trainable_layers = 30 # number of trainable layers at the top of the model; all other bottom layers will be frozen

train_dir = "../training_data/train"
test_dir  = "../training_data/validate"

output_name = "PurifyAI_GACC_MobileNetV2_{dim_img}_lr{lr}bs{bs}ep{ep}tl{tl}".format(dim_img=img_size, lr=learning_rate, bs=batch_size, ep=epochs, tl=trainable_layers)

def draw_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#%%
### Load model
model = models.load_model('../models/PurifyAI_GACC_MobileNetV2_224.h5')

#%%
### Load test images
img_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_img_generator = img_generator.flow_from_directory(
                        test_dir,
                        target_size = (img_size, img_size),
                        class_mode = 'categorical',
                        batch_size= batch_size,
                        interpolation = 'lanczos',
                        shuffle = False)

class_names = list(test_img_generator.class_indices.keys())
print("""Class names: {}""".format(class_names))

steps_test = test_img_generator.n // batch_size
test_classes = test_img_generator.classes[:steps_test*batch_size]
print("""Steps on test: {}""".format(steps_test))

#%%
### Evaluate model accuracy and loss
test_img_generator.reset()
results = model.evaluate_generator(test_img_generator, steps_test, workers=4)
print("Loss: ", "{0:.4f}".format(results[0]), "Accuracy: ", "{0:.4f}".format(results[1]))

#%%
### Produce confusion matrix
test_img_generator.reset()
predictions = model.predict_generator(test_img_generator, steps_test, workers=4)

# Convert the predicted classes from arrays to integers.
predicted_class_indices = np.argmax(predictions, axis=1)

# Get the confusion matrix using sklearn.
cfmx = confusion_matrix(y_true=test_classes,  # True class for test-set.
                        y_pred=predicted_class_indices)  # Predicted class.

draw_confusion_matrix(cfmx, class_names, (4,3))

#%%
### Detailed list of all filenames, predictions and scores
labels = (test_img_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions_labeled = [labels[k] for k in predicted_class_indices]

pd.set_option('display.max_rows', None)
cdf = pd.DataFrame({"Filename": test_img_generator.filenames[:len(test_classes)],
              "Prediction": predictions_labeled,
              "Benign": ["{0:.4f}".format(i[0]) for i in predictions],
              "Malign": ["{0:.4f}".format(i[1]) for i in predictions]})

output_file = '../data/'+'{name}.csv'.format(name=output_name)
cdf.to_csv(output_file, encoding='utf-8', index=False)