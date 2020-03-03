# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Provides utilities to preprocess images.

Training images are sampled using the optionally provided bounding boxes,
and subsequently cropped to the sampled bounding box. Images are additionally
flipped randomly, undergo colour distortion and then resized to the target 
output size (without aspect-ratio preservation).

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped.

Note that these steps are specific to Inception preprocessing.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf

DEFAULT_IMAGE_SIZE = 299
NUM_CHANNELS = 3
RESIZE_METHOD = tf.image.ResizeMethod.BICUBIC
CENTRAL_FRACTION = 0.875
_SHUFFLE_BUFFER = 10000


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           drop_remainder=False,
                           tf_data_experimental_slack=False):
    """Given a Dataset with raw records, return an iterator over the records.

    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for images/features.
      datasets_num_private_threads: Number of threads for a private
        threadpool created for all datasets computation.
      drop_remainder: A boolean indicates whether to drop the remainder of the
        batches. If True, the batch dimension will be static.
      tf_data_experimental_slack: Whether to enable tf.data's
        `experimental_slack` option.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """
    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
        options = tf.data.Options()
        options.experimental_threading.private_threadpool_size = (
            datasets_num_private_threads)
        dataset = dataset.with_options(options)
        logging.info(
            'datasets_num_private_threads: %s', datasets_num_private_threads)

    if is_training:
        # Shuffles records before repeating to respect epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        # Repeats the dataset for the number of epochs to train.
        dataset = dataset.repeat()

    # Parses the raw records into images and labels.
    dataset = dataset.map(
        lambda value: parse_record_fn(value, is_training, dtype),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.data.experimental.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_slack = tf_data_experimental_slack
    dataset = dataset.with_options(options)

    return dataset


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                   default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                  default_value=''),
    }
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in [
            'image/object/bbox/xmin', 'image/object/bbox/ymin',
            'image/object/bbox/xmax', 'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(serialized=example_serialized,
                                          features=feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

    return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
      raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
      is_training: A boolean denoting whether the input is for training.
      dtype: data type to use for images/features.

    Returns:
      Tuple with processed image tensor in a channel-last format and
      one-hot-encoded label tensor.
    """
    image_buffer, label, bbox = parse_example_proto(raw_record)

    image = preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=DEFAULT_IMAGE_SIZE,
        output_width=DEFAULT_IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        is_training=is_training)
    image = tf.cast(image, dtype)

    # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # Keras model.
    label = tf.cast(tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1,
                    dtype=tf.float32)
    return image, label


def get_parse_record_fn(use_keras_image_data_format=False):
    """Get a function for parsing the records, accounting for image format.

    This is useful by handling different types of Keras models. For instance,
    the current resnet_model.resnet50 input format is always channel-last,
    whereas the keras_applications mobilenet input format depends on
    tf.keras.backend.image_data_format(). We should set
    use_keras_image_data_format=False for the former and True for the latter.

    Args:
      use_keras_image_data_format: A boolean denoting whether data format is
        keras backend image data format. If False, the image format is
        channel-last.
        If True, the image format matches tf.keras.backend.image_data_format().

    Returns:
      Function to use for parsing the records.
    """
    def parse_record_fn(raw_record, is_training, dtype):
        image, label = parse_record(raw_record, is_training, dtype)
        if use_keras_image_data_format:
            if tf.keras.backend.image_data_format() == 'channels_first':
                image = tf.transpose(image, perm=[2, 0, 1])
        return image, label
    return parse_record_fn


def input_fn(is_training,
             filenames,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False,
             tf_data_experimental_slack=False,
             training_dataset_cache=False):
    """Input function which provides batches for train or eval.

    Args:
      is_training: A boolean denoting whether the input is for training.
      filenames: List of TFRecords file names containing image data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for images/features
      datasets_num_private_threads: Number of private threads for tf.data.
      parse_record_fn: Function to use for parsing the records.
      input_context: A `tf.distribute.InputContext` object passed in by
        `tf.distribute.Strategy`.
      drop_remainder: A boolean indicates whether to drop the remainder of the
        batches. If True, the batch dimension will be static.
      tf_data_experimental_slack: Whether to enable tf.data's
        `experimental_slack` option.
      training_dataset_cache: Whether to cache the training dataset on workers.
         Typically used to improve training performance when training data is
         in remote storage and can fit into worker memory.

    Returns:
      A dataset that can be used for iteration.
    """
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if input_context:
        logging.info(
            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
            input_context.input_pipeline_id, input_context.num_input_pipelines)
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=len(filenames))

    # Convert to individual records.
    # cycle_length = 10 means that up to 10 files will be read and deserialized in
    # parallel. You may want to increase this number if you have a large number of
    # CPU cores.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training and training_dataset_cache:
        # Improve training performance when training data is in remote storage and
        # can fit into worker memory.
        dataset = dataset.cache()

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record_fn,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        drop_remainder=drop_remainder,
        tf_data_experimental_slack=tf_data_experimental_slack,
    )


def _decode_jpeg(image_buffer):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
      image_buffer: scalar string Tensor.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image


def _distort_image(image, height, width, bbox, num_channels):
    """Distort one image for training a network.
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.
    Args:
      image: 3-D float Tensor of image
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      num_channels: Number of channels of image
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an allowed
    # range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    distorted_image = tf.image.resize(
        distorted_image, [height, width], method=RESIZE_METHOD)

    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, num_channels])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors.
    distorted_image = _distort_color(distorted_image)

    return distorted_image


def _eval_image(image, height, width):
    """Prepare one image for evaluation.
    Args:
      image: 3-D float Tensor
      height: integer
      width: integer
    Returns:
      3-D float Tensor of prepared image.
    """
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=CENTRAL_FRACTION)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, [height, width], method=RESIZE_METHOD)
    image = tf.squeeze(image, [0])

    return image


def _distort_color(image, thread_id=0):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather than adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
    Returns:
      color-distorted image
    """
    color_ordering = thread_id % 2

    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def preprocess_image(image_buffer, bbox, output_height, output_width,
                     num_channels, is_training=False):
    """Preprocesses the given image.

    Preprocessing includes decoding, cropping, and resizing for both training
    and eval images. Training preprocessing, however, introduces some random
    distortion of the image to improve accuracy.

    Args:
      image_buffer: scalar string Tensor representing the raw JPEG image buffer.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      num_channels: Integer depth of the image buffer for decoding.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.

    Returns:
      A preprocessed image.
    """
    image = _decode_jpeg(image_buffer)

    if is_training:
        # For training, we want to randomize some of the distortions.
        image = _distort_image(image, output_height,
                               output_width, bbox, num_channels)
    else:
        # For validation, we want to decode, resize, then just crop the middle.
        image = _eval_image(image, output_height, output_width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image
