# coding=utf-8
# Copyright 2020 The SimCLR Authors.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from google.cloud import storage

import functools
from absl import flags

import data_util as data_util
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def pad_to_batch(dataset, batch_size):
    """Pad Tensors to specified batch size.

    Args:
    dataset: An instance of tf.data.Dataset.
    batch_size: The number of samples per batch of input requested.

    Returns:
    An instance of tf.data.Dataset that yields the same Tensors with the same
    structure as the original padded to batch_size along the leading
    dimension.

    Raises:
    ValueError: If the dataset does not comprise any tensors; if a tensor
      yielded by the dataset has an unknown number of dimensions or is a
      scalar; or if it can be statically determined that tensors comprising
      a single dataset element will have different leading dimensions.
    """
    def _pad_to_batch(*args):
        """Given Tensors yielded by a Dataset, pads all to the batch size."""
        flat_args = tf.nest.flatten(args)

        for tensor in flat_args:
            if tensor.shape.ndims is None:
                raise ValueError(
                    'Unknown number of dimensions for tensor %s' % tensor.name)
            if tensor.shape.ndims == 0:
                raise ValueError('Tensor %s is a scalar.' % tensor.name)

        # This will throw if flat_args is empty. However, as of this writing,
        # tf.data.Dataset.map will throw first with an internal error, so we do
        # not check this case explicitly.
        first_tensor = flat_args[0]
        first_tensor_shape = tf.shape(first_tensor)
        first_tensor_batch_size = first_tensor_shape[0]
        difference = batch_size - first_tensor_batch_size

        for i, tensor in enumerate(flat_args):
            control_deps = []
            if i != 0:
                # Check that leading dimensions of this tensor matches the
                # first, either statically or dynamically. (If the first
                # dimensions of both tensors are statically known, the we have
                # to check the static shapes at graph construction time or else
                # we will never get to the dynamic assertion.)
                if (first_tensor.shape[:1].is_fully_defined() and
                        tensor.shape[:1].is_fully_defined()):
                    if first_tensor.shape[0] != tensor.shape[0]:
                        raise ValueError('Batch size of dataset tensors does '
                            'not match. %s has shape %s, but %s has shape %s' % (
                            first_tensor.name, first_tensor.shape,
                            tensor.name, tensor.shape))
                else:
                    curr_shape = tf.shape(tensor)
                    control_deps = [tf.Assert(
                        tf.equal(curr_shape[0], first_tensor_batch_size),
                        ['Batch size of dataset tensors %s and %s do not match'
                         'Shapes are' % (tensor.name, first_tensor.name),
                         curr_shape, first_tensor_shape])]

            with tf.control_dependencies(control_deps):
                # Pad to batch_size along leading dimension.
                flat_args[i] = tf.pad(tensor,
                    [[0, difference]] + [[0, 0]] * (tensor.shape.ndims - 1))
            flat_args[i].set_shape([batch_size] + tensor.shape.as_list()[1:])

        return tf.nest.pack_sequence_as(args, flat_args)

    return dataset.map(_pad_to_batch)


def build_multi_kingdom_input_fn(tot_num_classes,
                                 animal_num_classes,
                                 plant_num_classes,
                                 is_training):
    """
    Build input function for animal/plant, multikingdom model.
    """
    def _input_fn(params):
        preprocess_fn_finetune = get_preprocess_fn(is_training,
                                                   is_pretrain=False)
        client = storage.Client()
        B_NAME = FLAGS.gcs_bucket_name
        filenames = []

        with tf.io.gfile.GFile(FLAGS.labels_path, 'r') as f:
            all_labels = f.read().splitlines()
        with tf.io.gfile.GFile(FLAGS.animal_labels_path, 'r') as f:
            animals_labels = f.read().splitlines()
        with tf.io.gfile.GFile(FLAGS.plant_labels_path, 'r') as f:
            plants_labels = f.read().splitlines()

        assert len(all_labels) == len(animals_labels) + len(plants_labels)
        assert tot_num_classes == len(all_labels)
        assert animal_num_classes == len(animals_labels)
        assert plant_num_classes == len(plants_labels)

        # label_idx = 0
        # all_label_dict = dict()
        # for l in all_labels:
        #     all_label_dict[label_idx] = l
        #     label_idx += 1
        #
        # animals_label_idx = 0
        # animals_label_dict = dict()
        # for l in animals_labels:
        #     animals_label_dict[l] = animals_label_idx
        #     animals_label_idx += 1
        #
        # plants_label_idx = 0
        # plants_label_dict = dict()
        # for l in plants_labels:
        #     plants_label_dict[l] = plants_label_idx
        #     plants_label_idx += 1
        #
        # animals_labels = set(animals_labels)
        # plants_labels = set(plants_labels)

        def map_fn(example_proto):
            img_feat_desc = {
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
                'image/class/animal_label': tf.io.FixedLenFeature([], tf.int64),
                'image/class/plant_label': tf.io.FixedLenFeature([], tf.int64),
                'image/mask/animal_mask': tf.io.FixedLenFeature([], tf.float32),
                'image/mask/plant_mask': tf.io.FixedLenFeature([], tf.float32),
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
            }
            feat = tf.io.parse_single_example(example_proto, img_feat_desc)
            image = tf.image.decode_jpeg(feat['image/encoded'], channels=3)
            label = feat['image/class/label']
            animal_label = feat['image/class/animal_label']
            plant_label = feat['image/class/plant_label']
            animal_mask = feat['image/mask/animal_mask']
            plant_mask = feat['image/mask/plant_mask']

            # if all_label_dict[label] in plants_labels:
            #     plant_label = plants_label_dict[all_label_dict[label]]
            #     plant_mask = 1.0
            # else:
            #     animal_label = animals_label_dict[all_label_dict[label]]
            #     animal_mask = 1.0

            if FLAGS.mode == 'predict':
                image = preprocess_fn_finetune(image)
                label = tf.one_hot(label, tot_num_classes)
                animal_label = tf.one_hot(animal_label, animal_num_classes)
                plant_label = tf.one_hot(plant_label, plant_num_classes)
                return image, {'label': label,
                               'animal_label': animal_label,
                               'plant_label': plant_label,
                               'mask': 1.0,
                               'animal_mask': animal_mask,
                               'plant_mask': plant_mask}

            if FLAGS.train_mode == 'pretrain':
                print('pretrain mode not defined.')
            else:
                image = preprocess_fn_finetune(image)
                label = tf.one_hot(label, tot_num_classes)
                animal_label = tf.one_hot(animal_label, animal_num_classes)
                plant_label = tf.one_hot(plant_label, plant_num_classes)
            return image, label, animal_label, plant_label, 1.0, animal_mask,\
                plant_mask

        # Returns a tf.data.Dataset object
        if is_training:
            F_PATH = FLAGS.train_path
            for blob in client.list_blobs(B_NAME, prefix=F_PATH):
                filenames.append('gs://' + B_NAME + '/' + blob.name)
        else:
            F_PATH = FLAGS.val_path
            for blob in client.list_blobs(B_NAME, prefix=F_PATH):
                filenames.append('gs://' + B_NAME + '/' + blob.name)

        dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        if FLAGS.cache_dataset:
            dataset = dataset.cache()
        if is_training:
            buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
            dataset = dataset.shuffle(params['batch_size'] * buffer_multiplier)
            dataset = dataset.repeat(-1)
        dataset = dataset.map(map_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(params['batch_size'],
                                drop_remainder=is_training)

        if FLAGS.mode == 'predict':
            return dataset

        dataset = pad_to_batch(dataset, params['batch_size'])
        # Mask is always 1.0
        image, label, animal_label, plant_label, mask, animal_mask,\
            plant_mask = tf.data.make_one_shot_iterator(dataset).get_next()

        return image, {'label': label,
                       'animal_label': animal_label,
                       'plant_label': plant_label,
                       'mask': mask,
                       'animal_mask': animal_mask,
                       'plant_mask': plant_mask}

    return _input_fn



def build_input_fn(n_classes, is_training):
    """Build input function.

    Args:
    builder: TFDS builder for specified dataset.
    is_training: Whether to build in training mode.

    Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
    """
    def _input_fn(params):
        """Inner input function."""
        preprocess_fn_pretrain = get_preprocess_fn(is_training,
                                                   is_pretrain=True)
        preprocess_fn_finetune = get_preprocess_fn(is_training,
                                                   is_pretrain=False)
        num_classes = n_classes

        client = storage.Client()
        B_NAME = FLAGS.gcs_bucket_name
        filenames = []

        def map_fn(example_proto):
            """Produces multiple transformations of the same batch."""
            img_feat_desc = {
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
            }
            feat = tf.io.parse_single_example(example_proto, img_feat_desc)
            image = tf.image.decode_jpeg(feat['image/encoded'], channels=3)
            label = feat['image/class/label']

            if FLAGS.mode == 'predict':
                image = preprocess_fn_finetune(image)
                label = tf.one_hot(label, num_classes)
                return image, {'labels': label, 'mask': 1.0}

            if FLAGS.train_mode == 'pretrain':
                xs = []
                for _ in range(2):  # Two transformations
                    xs.append(preprocess_fn_pretrain(image))
                image = tf.concat(xs, -1)
                label = tf.zeros([num_classes])
            else:
                image = preprocess_fn_finetune(image)
                label = tf.one_hot(label, num_classes)
            return image, label, 1.0

        # Returns a tf.data.Dataset object
        if is_training:
            F_PATH = FLAGS.train_path
            for blob in client.list_blobs(B_NAME, prefix=F_PATH):
                filenames.append('gs://' + B_NAME + '/' + blob.name)
        else:
            F_PATH = FLAGS.val_path
            for blob in client.list_blobs(B_NAME, prefix=F_PATH):
                filenames.append('gs://' + B_NAME + '/' + blob.name)

        dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        if FLAGS.cache_dataset:
            dataset = dataset.cache()
        if is_training:
            buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
            dataset = dataset.shuffle(params['batch_size'] * buffer_multiplier)
            dataset = dataset.repeat(-1)
        dataset = dataset.map(map_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(params['batch_size'], drop_remainder=is_training)

        if FLAGS.mode == 'predict':
            return dataset

        dataset = pad_to_batch(dataset, params['batch_size'])
        # Mask is always 1.0
        images, labels, mask = tf.data.make_one_shot_iterator(dataset).get_next()

        return images, {'labels': labels, 'mask': mask}
    return _input_fn


def get_preprocess_fn(is_training, is_pretrain):
    """Get function that accepts an image and returns a preprocessed image."""
    # Disable test cropping for small images (e.g. CIFAR)
    if FLAGS.image_size <= 32:
        test_crop = False
    else:
        test_crop = True
    return functools.partial(data_util.preprocess_image,
                             height=FLAGS.image_size,
                             width=FLAGS.image_size,
                             is_training=is_training,
                             color_distort=is_pretrain,
                             test_crop=test_crop)
