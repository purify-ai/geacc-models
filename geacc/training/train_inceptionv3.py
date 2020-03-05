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

# %%
import os
from time import time

import tensorflow as tf
from . import project_path
from . import image_preprocessing
from . import distribution_utils

# Hyperparameters
IMG_SIZE = 299
OUTPUT_CLASSES_NUM = 3

HPARAMS = {
    'optimizer':        'adam',  # 'sgd' or 'adam'
    'momentum':         0.9,     # for SGD
    'learning_rate':    0.002,   # for Adam
    'batch_size':       1024,
    'total_epochs':     30,
    'frozen_layer_num': 168,

    # temporarily hardcoded, move to dataset.info
    'class_names':              ['benign', 'explicit', 'suggestive'],
    'train_image_files':        8000 * 3,
    'validate_image_files':     1000 * 3,
    'test_image_files':         1000 * 3,
    'train_tfrecord_files':     8,
    'validate_tfrecord_files':  1,
    'test_tfrecord_files':      1,
}

# Other Consts
OUTPUT_MODEL_PREFIX = f"Geacc_InceptionV3_{int(time())}"
DATASET_PATH = 'data/dataset'
OUTPUT_PATH = 'data/models'
TENSORBOARD_PATH = False
TPU_ADDRESS = False


# %%
def get_filenames(is_training, data_dir):
    """Get TFRecord filenames for dataset."""
    if is_training:
        filenames = [
            os.path.join(data_dir, 'train-{:05d}-of-{:05d}'.format(i, HPARAMS['train_tfrecord_files'])) for i in range(HPARAMS['train_tfrecord_files'])]
    else:
        filenames = [
            os.path.join(data_dir, 'validation-{:05d}-of-{:05d}'.format(i, HPARAMS['validate_tfrecord_files'])) for i in range(HPARAMS['validate_image_files'])]

    if TPU_ADDRESS and not distribution_utils.tpu_compatible_files(filenames):
        raise Exception("TPU requires files stored in Google Cloud Storage (GCS) buckets.")

    return filenames


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
        optimizer = tf.keras.optimizers.SGD(momentum=hparams['momentum'])
    elif hparams['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=hparams['learning_rate'])
    else:
        raise ValueError('Invalid value of optimizer: %s' %
                         hparams['optimizer'])
    return optimizer


def init_callbacks():
    """Setup callbacks."""
    use_callbacks = []

    # Checkpoint
    checkpoint_file = os.path.join(
        OUTPUT_PATH, OUTPUT_MODEL_PREFIX + "_ep{epoch:02d}_vl{val_loss:.2f}.tf")
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file, monitor='val_loss', verbose=1, save_best_only=False)
    use_callbacks.append(checkpointer)

    # Early stopping
    # early_stop = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', min_delta=0.001, patience=10)
    # use_callbacks.append(early_stop)

    # learning rate schedule
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        step_decay_schedule)
    use_callbacks.append(lr_scheduler)

    # Tensorboard logs
    if (TENSORBOARD_PATH):
        tb_logs_dir = os.path.join(TENSORBOARD_PATH, OUTPUT_MODEL_PREFIX)
        print('TensorBoard events:', tb_logs_dir)

        # Capture hyperparameters in Tensorboard
        with tf.summary.create_file_writer(tb_logs_dir + "/hparams").as_default():
            tensor = tf.stack([tf.convert_to_tensor([k, str(v)])
                                    for k, v in HPARAMS.items()])
            tf.summary.text("Hyperparameters", tensor, step=0)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tb_logs_dir)
        use_callbacks.append(tensorboard)

    return use_callbacks


# %%
# Build InceptionV3 model
def build_model():
    tf.keras.backend.clear_session()

    pretrained_model = tf.keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze bottom layers
    for layer in pretrained_model.layers[:HPARAMS['frozen_layer_num']]:
        layer.trainable = False

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4096),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(OUTPUT_CLASSES_NUM, activation='softmax')
    ])

    return model


def optimize_performance():
    # Use mixed precision when available
    if TPU_ADDRESS:
        policy = 'mixed_bfloat16'
    elif GPU_NUM > 0:
        policy = 'mixed_float16'
    else:
        policy = 'float32'

    print(f"Setting mixed precision policy to {policy}.")
    tf.keras.mixed_precision.experimental.set_policy(policy)
    HPARAMS['mixed_precision_policy'] = policy

# %%
# Prepare images for training
def load_datasets():
    train_input_dataset = image_preprocessing.input_fn(
        is_training = True,
        filenames   = get_filenames(is_training=True, data_dir=DATASET_PATH),
        batch_size  = HPARAMS['batch_size'],
        num_epochs  = HPARAMS['total_epochs'],
        parse_record_fn = image_preprocessing.get_parse_record_fn(one_hot_encoding_class_num=OUTPUT_CLASSES_NUM),
        # datasets_num_private_threads=None,
        # dtype=tf.float32,
        # drop_remainder=False,
        # tf_data_experimental_slack=False,
        # training_dataset_cache=False,
    )

    validate_input_dataset = image_preprocessing.input_fn(
        is_training = False,
        filenames   = get_filenames(is_training=True, data_dir=DATASET_PATH),
        batch_size  = HPARAMS['batch_size'],
        num_epochs  = HPARAMS['total_epochs'],
        parse_record_fn = image_preprocessing.get_parse_record_fn(one_hot_encoding_class_num=OUTPUT_CLASSES_NUM),
    )

    class_names = HPARAMS['class_names']
    print(f"Class names: {class_names}")

    steps_train = HPARAMS['train_image_files'] // HPARAMS['batch_size']
    print(f"Steps on train: {steps_train}")

    steps_validate = HPARAMS['validate_image_files'] // HPARAMS['batch_size']
    print(f"Steps on validation: {steps_validate}")

    return [train_input_dataset, validate_input_dataset, steps_train, steps_validate]


def load_checkpoint(checkpoint_file, model):
    if os.path.exists(checkpoint_file):
        print("Resuming from checkpoint: ", checkpoint_file)
        model.load_weights(checkpoint_file)

        # Finding the epoch index from which we are resuming
        # initial_epoch = get_init_epoch(checkpoint_path)

        # Calculating the correct value of count
        # count = initial_epoch*batches_per_epoch


def train(dataset_path='data/dataset',
          model_path='data/model',
          tb_path=False,
          distribution_strategy='',
          tpu_address='',
          gpu_num=0):
    """Run InceptionV3 training and eval loop using native Keras APIs.

    Args:
        dataset_path: Path to TFRecord data files for input
        model_path: Path for model output
        tb_path: TensorBoard log dir
        distribution_strategy: ...
        tpu_address: TPU address. TPU will not be used if not set.
        gpu_num: Number of GPUs.
    Raises:
        ValueError: If fp16 is passed as it is not currently supported.
        NotImplementedError: If some features are not currently supported.
    Returns:
        Dictionary of training and eval stats.
    """

    global DATASET_PATH
    DATASET_PATH = dataset_path
    global OUTPUT_PATH
    OUTPUT_PATH = model_path
    global TENSORBOARD_PATH
    TENSORBOARD_PATH = tb_path
    global TPU_ADDRESS
    TPU_ADDRESS = tpu_address
    global GPU_NUM
    GPU_NUM = gpu_num

    optimize_performance()

    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=distribution_strategy,
        num_gpus=gpu_num,
        tpu_address=tpu_address)

    strategy_scope = distribution_utils.get_strategy_scope(strategy)

    # Get training/validation datasets
    train_input_dataset, validate_input_dataset, steps_train, steps_validate = load_datasets()

    with strategy_scope:
        model = build_model()
        model.summary()

        model.compile(
            optimizer=get_optimizer(HPARAMS),
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=[tf.metrics.CategoricalAccuracy(), tf.metrics.AUC(), tf.metrics.Precision(), tf.metrics.Recall()]
        )

    print('Starting training')

    model.fit(
        train_input_dataset,
        steps_per_epoch=steps_train,
        epochs=HPARAMS['total_epochs'],
        validation_data=validate_input_dataset,
        validation_steps=steps_validate,
        callbacks=init_callbacks(),
        #verbose=2
    )

    # Save final model
    model.save(os.path.join(OUTPUT_PATH, OUTPUT_MODEL_PREFIX + "_final.tf"))
