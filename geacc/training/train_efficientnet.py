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

# Fine-tuning ImageNet trained EfficientNet-B3

#%%
import os
import argparse
from pathlib import Path
from time import time

from training import preprocess_crop
from keras import initializers, regularizers, losses, callbacks, layers, backend, models
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
import efficientnet.keras as efnet
from efficientnet.keras import preprocess_input

# Hyperparameters
IMG_SIZE = 300
OUTPUT_CLASSES_NUM = 3
BATCH_SIZE = 128
TOTAL_EPOCHS = 100

# Other Consts
MODEL_TYPE = "EfficientNetB3"
OUTPUT_MODEL_PREFIX = f"Geacc_{MODEL_TYPE}_{int(time())}"
OUTPUT_MODEL_NAME = OUTPUT_MODEL_PREFIX + f"_{IMG_SIZE}x{IMG_SIZE}_bs{BATCH_SIZE}"

DATASET_PATH = 'data/dataset'
OUTPUT_PATH = 'data/models'
TENSORBOARD_PATH = False


#%%
def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", metavar="PATH", required=True,
                        help="Path to the dataset containing training and validation data")
    parser.add_argument("-m", "--models", metavar="PATH", required=True,
                        help="Path to store trained intermediate and final models")
    parser.add_argument("-t", "--tensorboard", metavar="PATH", required=False,
                        help="Path to store TensorBoard logs")

    args = parser.parse_args()

    global DATASET_PATH; DATASET_PATH = args.dataset
    global OUTPUT_PATH; OUTPUT_PATH = args.models
    global TENSORBOARD_PATH; TENSORBOARD_PATH = args.tensorboard


# Slow down training deeper into dataset
def step_decay_schedule(epoch):
    if epoch < 2:
        # Warmup model first
        return .0000032
    elif epoch < 8:
        return .01
    elif epoch < 16:
        return .002
    elif epoch < 32:
        return .0004
    elif epoch < 64:
        return .00008
    elif epoch < 96:
        return .000016
    else:
        return .0000009


# Setup callbacks
def init_callbacks():
    use_callbacks = []

    # Checkpoint
    checkpoint_file = os.path.join(OUTPUT_PATH, OUTPUT_MODEL_NAME + "_ep{epoch:02d}_vl{val_loss:.2f}.h5")
    checkpointer = callbacks.ModelCheckpoint(
        filepath=checkpoint_file, monitor='val_loss', verbose=1, save_best_only=False)
    use_callbacks.append(checkpointer)

    # Early stopping
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
    #use_callbacks.append(early_stop)

    # learning rate schedule
    lr_scheduler = callbacks.LearningRateScheduler(step_decay_schedule)
    use_callbacks.append(lr_scheduler)

    # Tensorboard logs
    if (TENSORBOARD_PATH):
        tb_logs_dir = os.path.join(TENSORBOARD_PATH, OUTPUT_MODEL_PREFIX)
        print('TensorBoard events:', tb_logs_dir)
        tensorboard = callbacks.TensorBoard(log_dir=tb_logs_dir)
        use_callbacks.append(tensorboard)

    return use_callbacks


#%%
# Build model
def build_model():
    backend.clear_session()

    base_model = efnet.EfficientNetB3(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze bottom layers
    is_trainable = False
    for layer in base_model.layers:
        if layer.name == 'multiply_16': is_trainable = True
        layer.trainable = is_trainable

    # Construct top layer replacement
    x = base_model.output
    x = layers.GlobalMaxPooling2D(name="gap")(x)
    # x = layers.GlobalAveragePooling2D()(x)  # Avg instead of Max
    # x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.2, name="dropout_out")(x)
    x = layers.Dense(256, activation='relu', name="fc1")(x)

    output_tensor = layers.Dense(OUTPUT_CLASSES_NUM, activation='softmax', name="fc_out")(x)
    model = models.Model(inputs = base_model.input, outputs=output_tensor)

    return model


#%%
### Prepare images for training
def create_generators():
    train_dir = os.path.join(DATASET_PATH, 'train')
    validate_dir = os.path.join(DATASET_PATH, 'validate')

    print('Train image generator')

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=[0.8, 1],
        channel_shift_range=20,
        horizontal_flip=True,
        #fill_mode='nearest',
        preprocessing_function=preprocess_input
    )

    train_img_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        interpolation = 'lanczos:random',
        shuffle = True
    )

    print('Validation image generator')

    # Validation data should not be augmented
    validate_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    validate_img_generator = validate_datagen.flow_from_directory(
        validate_dir,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        interpolation = 'lanczos:center',
        shuffle = False
    )

    train_classes = train_img_generator.classes
    validate_classes = validate_img_generator.classes

    class_names = list(train_img_generator.class_indices.keys())
    print(f"Class names: {class_names}")

    steps_train = train_img_generator.n // BATCH_SIZE
    print(f"Steps on train: {steps_train}")

    steps_validate = validate_img_generator.n // BATCH_SIZE
    print(f"Steps on validation: {steps_validate}")

    return [train_img_generator, validate_img_generator, steps_train, steps_validate]


def load_checkpoint(checkpoint_file, model):
    
    if os.path.exists(checkpoint_file):
        print ("Resuming from checkpoint: ", checkpoint_file)
        model.load_weights(checkpoint_file)

        # Finding the epoch index from which we are resuming
        #initial_epoch = get_init_epoch(checkpoint_path)

        # Calculating the correct value of count
        #count = initial_epoch*batches_per_epoch


#%%
### MAIN
input_arguments()

model = build_model()
#model.summary()

model.compile(
    optimizer = SGD(momentum=0.9, nesterov=True),
    #optimizer = Adam(lr=0.0005),
    loss = losses.categorical_crossentropy,
    metrics = ['accuracy']
)

#%%
# Get training/validation data via generators
train_img_generator, validate_img_generator, steps_train, steps_validate = create_generators()

#%%
print('Available GPUs:', backend.tensorflow_backend._get_available_gpus())
print('Starting training')

model.fit_generator(
    train_img_generator,
    steps_per_epoch = steps_train,
    epochs = TOTAL_EPOCHS,
    validation_data = validate_img_generator,
    validation_steps = steps_validate,
    callbacks = init_callbacks()
    # initial_epoch = ... restart from checkpoint
    # workers=6, use_multiprocessing=True
 )

### Save final model
model.save(os.path.join(OUTPUT_PATH, OUTPUT_MODEL_NAME + "_final.h5"))