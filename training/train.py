# Copyright 2019 Purify Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fine-tuning ImageNet trained MobileNetV2 with GACC dataset

#%%
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, callbacks, backend
from keras.optimizers import Adam
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter

### Hyperparameters
output_classes = 2
learning_rate = 4e-5  #0.00004
img_size = 224 # img width and height
batch_size = 128
epochs = 60
trainable_layers = 30 # number of trainable layers at the top of the model; all other bottom layers will be frozen

train_dir = "../training_data/train"
test_dir  = "../training_data/validate"

output_name = "PurifyAI_GACC_MobileNetV2_{dim_img}_lr{lr}bs{bs}ep{ep}tl{tl}".format(dim_img=img_size, lr=learning_rate, bs=batch_size, ep=epochs, tl=trainable_layers)
tensorboard_logs = "./tb_logs/"

print('Available GPUs:', backend.tensorflow_backend._get_available_gpus())
print('TensorBoard events:', tensorboard_logs)

#%%
### Prepare images for training
print('Train image generator')
img_generator_augment = ImageDataGenerator(
                        rotation_range=23,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        horizontal_flip=True,
                        preprocessing_function=preprocess_input)

train_img_generator = img_generator_augment.flow_from_directory(
                        train_dir,
                        target_size = (img_size, img_size),
                        batch_size = batch_size,
                        class_mode = 'categorical',
                        interpolation = 'lanczos',
                        shuffle = True)

print('Test image generator')
img_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_img_generator = img_generator.flow_from_directory(
                        test_dir,
                        target_size = (img_size, img_size),
                        batch_size= batch_size,
                        class_mode = 'categorical',
                        interpolation = 'lanczos',
                        shuffle = False)

train_classes = train_img_generator.classes
test_classes = test_img_generator.classes

class_names = list(train_img_generator.class_indices.keys())
print("""Class names: {}""".format(class_names))

steps_train = train_img_generator.n // batch_size
print("""Steps on train: {}""".format(steps_train))

steps_test = test_img_generator.n // batch_size
print("""Steps on test: {}""".format(steps_test))

#%%
### Calculate class weights for balancing
#counter = Counter(train_classes)
#max_val = float(max(counter.values()))
#class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
#print("Class weights:", class_weights)

#%%
### Build and compile the model
def build_model():
    """ Build new model through the following steps:
        1. Load ImageNet trained MobileNetV2 without fully-connected layer at the top of the network
        2. Freeze all layers except the top M layers. Top M layers will be trainable.
        3. Add final fully-connected (Dense) layer
    """
    input_tensor = layers.Input(shape=(img_size, img_size, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(img_size, img_size, 3),
        pooling='avg'
    )

    # Only top M layers are trainable
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    output_tensor = layers.Dense(output_classes, activation='softmax')(base_model.output)
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model

model = build_model()
#model.summary()

model.compile(optimizer=Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

#%% 
### Train model
early_stop  = callbacks.EarlyStopping(monitor = 'val_loss', min_delta=0.01, patience=10)
tensorboard = callbacks.TensorBoard(log_dir=tensorboard_logs)

checkpoint_file = '../models/' + output_name + "_{epoch:02d}_{val_loss:.2f}.h5"
checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_file, verbose=1, save_best_only=True)

model.fit_generator(train_img_generator,
             steps_per_epoch = steps_train,
             epochs = epochs,
             validation_data = test_img_generator,
             validation_steps = steps_test,
             #class_weight = class_weights,
             callbacks=[tensorboard, checkpointer])

#%%
### Save final model
output_file = '../models/'+'{name}_final.h5'.format(name=output_name)
model.save(output_file)