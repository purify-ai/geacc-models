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

# Fine-tuning ImageNet trained InceptionV3

#%%
import os
import argparse
from pathlib import Path
from time import time

import preprocess_crop
import tensorflow as tf
from keras import initializers, regularizers, losses, callbacks, layers, backend, models
from keras.applications import InceptionV3
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
import extra_keras_metrics

# Hyperparameters
IMG_SIZE = 299
OUTPUT_CLASSES_NUM = 3

HPARAMS = {
    'optimizer':        'sgd', # or 'adam'
    'momentum':         0.9, # for SGD
    'learning_rate':    0.0005, # for Adam
    'batch_size':       128,
    'total_epochs':     1,
    'frozen_layer_num': 168
}

# Other Consts
OUTPUT_MODEL_PREFIX = f"Geacc_InceptionV3_{int(time())}"
OUTPUT_MODEL_NAME = OUTPUT_MODEL_PREFIX + f"_{IMG_SIZE}x{IMG_SIZE}_bs{HPARAMS['batch_size']}"

DATASET_PATH = 'data/dataset'
OUTPUT_PATH = 'data/models'
TENSORBOARD_PATH = False


#%%
class LoggingTensorBoard(callbacks.TensorBoard):    

    def __init__(self, log_dir, hparams, **kwargs):
        super(LoggingTensorBoard, self).__init__(log_dir, **kwargs)

        self.hparams = hparams

    def on_train_begin(self, logs=None):
        callbacks.TensorBoard.on_train_begin(self, logs=logs)

        tensor = tf.stack([tf.convert_to_tensor([k, str(v)]) for k, v in self.hparams.items()])
        text_summary = tf.summary.text("Hyperparameters", tensor)

        with tf.Session() as sess:
            s = sess.run(text_summary)
            self.writer.add_summary(s)
            self.writer.flush()

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

def get_optimizer(hparams):
  """Returns optimizer.

  Args:
    hparams: hyper parameters.

  Raises:
    ValueError: if type of optimizer specified in hparams is incorrect.

  Returns:
    Instance of optimizer class.
  """
  if hparams['optimizer'] == 'sgd':
    optimizer = SGD(momentum=hparams['momentum'])
  elif hparams['optimizer'] == 'adam':
    optimizer = Adam(lr=hparams['learning_rate'])
  else:
    raise ValueError('Invalid value of optimizer: %s' % hparams['optimizer'])
  return optimizer

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
        tensorboard = LoggingTensorBoard(log_dir=tb_logs_dir, hparams=HPARAMS)
        use_callbacks.append(tensorboard)

    return use_callbacks


#%%
# Build InceptionV3 model
def build_model():
    backend.clear_session()

    base_model = InceptionV3(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze bottom layers
    for layer in base_model.layers[:HPARAMS['frozen_layer_num']]:
        layer.trainable = False

    # Construct top layer replacement
    x = base_model.output

    # Alternative version 1
    #x = layers.AveragePooling2D(pool_size=(8, 8))(x)
    #x = layers.Dropout(0.4)(x)
    #x = layers.Flatten()(x)
    #x = layers.Dense(256, activation='relu', kernel_initializer=initializers.he_normal(seed=None), kernel_regularizer=regularizers.l2(0.0005))(x)
    #x = layers.Dropout(0.5)(x)
    # Essential to have another layer for better accuracy
    #x = layers.Dense(128, activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
    #x = layers.Dropout(0.25)(x)

    # Alternative version 2
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4096)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(.5)(x)

    output_tensor = layers.Dense(OUTPUT_CLASSES_NUM, activation='softmax')(x)
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
        batch_size  = HPARAMS['batch_size'],
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
        batch_size  = HPARAMS['batch_size'],
        class_mode  = 'categorical',
        interpolation = 'lanczos:center',
        shuffle = False
    )

    train_classes = train_img_generator.classes
    validate_classes = validate_img_generator.classes

    class_names = list(train_img_generator.class_indices.keys())
    print(f"Class names: {class_names}")

    steps_train = train_img_generator.n // HPARAMS['batch_size']
    print(f"Steps on train: {steps_train}")

    steps_validate = validate_img_generator.n // HPARAMS['batch_size']
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
    optimizer = get_optimizer(HPARAMS),
    loss = losses.categorical_crossentropy,
    metrics = ['accuracy', "auroc", "auprc"]
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
    epochs = HPARAMS['total_epochs'],
    validation_data = validate_img_generator,
    validation_steps = steps_validate,
    callbacks = init_callbacks()
    # initial_epoch = ... restart from checkpoint
    # workers=6, use_multiprocessing=True
 )

### Save final model
model.save(os.path.join(OUTPUT_PATH, OUTPUT_MODEL_NAME + "_final.h5"))