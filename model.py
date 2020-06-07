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
"""Model specification for SimCLR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import data_util as data_util
from lars_optimizer import LARSOptimizer
import model_util as model_util
import objective as obj_lib

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

FLAGS = flags.FLAGS


def build_multi_kingdom_model_fn(model, animal_num_classes, plant_num_classes,
                                 num_train_examples):
    """Build multihead model for plant and animal kingdom"""
    def model_fn(features, labels, mode, params=None):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Check training mode
        if FLAGS.train_mode == 'pretrain':
            raise ValueError('Pretraining not possible in multihead config,'
                             'Set train_mode to finetune')
        elif FLAGS.train_mode == 'finetune':
            # Base network forward pass.
            with tf.variable_scope('base_model'):
                if FLAGS.train_mode == 'finetune'\
                 and FLAGS.fine_tune_after_block >= 4:
                    # Finetune just supervised (linear) head will not
                    # update BN stats.
                    model_train_mode = False
                else:
                    model_train_mode = is_training
                hiddens = model(features, is_training=model_train_mode)

            logits_animal = model_util.supervised_head(hiddens,
                                                       animal_num_classes,
                                                       is_training)
            obj_lib.add_supervised_loss(labels=labels['animal_label'],
                                        logits=logits_animal,
                                        weights=labels['animal_mask'])

            logits_plant = model_util.supervised_head(hiddens,
                                                      plant_num_classes,
                                                      is_training)
            obj_lib.add_supervised_loss(labels=labels['plant_label'],
                                        logits=logits_plant,
                                        weights=labels['plant_label'])
            model_util.add_weight_decay(adjust_per_optimizer=True)
            loss = tf.losses.get_total_loss()

            collection_prefix = 'trainable_variables_inblock_'
            variables_to_train = []
            for j in range(FLAGS.fine_tune_after_block + 1, 6):
                variables_to_train += tf.get_collection(collection_prefix
                                                        + str(j))
            assert variables_to_train, 'variables_to_train shouldn\'t be empty'
            tf.logging.info('===============Variables to train (begin)===============')
            tf.logging.info(variables_to_train)
            tf.logging.info('================Variables to train (end)================')

            learning_rate = model_util.learning_rate_schedule(
                FLAGS.learning_rate, num_train_examples)

            if is_training:
                if FLAGS.train_summary_steps > 0:
                    summary_writer = tf2.summary.create_file_writer(FLAGS.model_dir)
                    with tf.control_dependencies([summary_writer.init()]):
                        with summary_writer.as_default():
                            should_record = tf.math.equal(tf.math.floormod(
                                tf.train.get_global_step(),
                                FLAGS.train_summary_steps), 0)
                        with tf.summary.record_if(should_record):
                            label_acc_animal = tf.equal(
                                tf.argmax(labels['animal_label'], 1),
                                tf.argmax(logits_animal, axis=1))
                            label_acc_plant = tf.equal(
                                tf.argmax(labels['plant_label'], 1),
                                tf.argmax(logits_plant, axis=1))

                            label_acc = tf.math.logical_or(label_acc_animal,
                                                           label_acc_plant)
                            label_acc = tf.reduce_mean(tf.cast(label_acc,
                                                               tf.float32))
                            tf2.summary.scalar(
                                'train_label_accuracy',
                                label_acc,
                                step=tf.train.get_global_step())
                            tf2.summary.scalar(
                                'learning_rate', learning_rate,
                                step=tf.train.get_global_step())
                if FLAGS.optimizer == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(
                        learning_rate, FLAGS.momentum, use_nesterov=True)
                elif FLAGS.optimizer == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                elif FLAGS.optimizer == 'lars':
                    optimizer = LARSOptimizer(
                        learning_rate, momentum=FLAGS.momentum,
                        weight_decay=FLAGS.weight_decay,
                        exclude_from_weight_decay=['batch_normalization',
                                                   'bias'])
                else:
                    raise ValueError('Unknown optimizer {}'.format(
                                FLAGS.optimizer))

                if FLAGS.use_tpu:
                    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
                control_deps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                if FLAGS.train_summary_steps > 0:
                    control_deps.extend(tf.summary.all_v2_summary_ops())
                with tf.control_dependencies(control_deps):
                    train_op = optimizer.minimize(
                        loss, global_step=tf.train.get_or_create_global_step(),
                        var_list=variables_to_train)

                if FLAGS.checkpoint:
                    def scaffold_fn():
                        """
                        Scaffold function to restore non-logits vars from
                        checkpoint.
                        """
                        tf.train.init_from_checkpoint(
                            FLAGS.checkpoint,
                            {v.op.name: v.op.name for v in tf.global_variables(
                                FLAGS.variable_schema)})

                        if FLAGS.zero_init_logits_layer:
                            # Init op that initializes output layer parameters
                            # to zeros.
                            output_layer_parameters = [
                                var for var in tf.trainable_variables()
                                if var.name.startswith('head_supervised')]
                            tf.logging.info(
                                'Initializing output layer parameters %s to 0',
                                [x.op.name for x in output_layer_parameters])
                            with tf.control_dependencies([tf.global_variables_initializer()]):
                                init_op = tf.group([
                                    tf.assign(x, tf.zeros_like(x))
                                    for x in output_layer_parameters])
                            return tf.train.Scaffold(init_op=init_op)
                        else:
                            return tf.train.Scaffold()
                else:
                    scaffold_fn = None

                return tf.estimator.tpu.TPUEstimatorSpec(
                    mode=mode, train_op=train_op, loss=loss,
                    scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            _, animal_top_5 = tf.nn.top_k(logits_animal, k=5)
            _, plant_top_5 = tf.nn.top_k(logits_plant, k=5)

            predictions = {
                'label': tf.argmax(labels['label'], 1),
                'animal_label': tf.argmax(labels['animal_label'], 1),
                'plant_label': tf.argmax(labels['plant_label'], 1),
                'animal_top_5': animal_top_5,
                'plant_top_5': plant_top_5,
            }

            return tf.estimator.tpu.TPUEstimatorSpec(mode=mode,
                                                     predictions=predictions)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(logits_animal, logits_plant, labels_animal,
                          labels_plant, mask_animal, mask_plant, **kws):
                metrics['label_animal_top_1_accuracy'] = tf.metrics.accuracy(
                    tf.argmax(labels_animal, 1),
                    tf.argmax(logits_animal, axis=1),
                    weights=mask_animal)
                metrics['label_animal_top_5_accuracy'] = tf.metrics.recall_at_k(
                    tf.argmax(labels_animal, 1),
                    logits_animal,
                    k=5,
                    weights=mask_animal)
                metrics['label_plant_top_1_accuracy'] = tf.metrics.accuracy(
                    tf.argmax(labels_plant, 1),
                    tf.argmax(logits_plant, axis=1),
                    weights=mask_plant)
                metrics['label_plant_top_5_accuracy'] = tf.metrics.recall_at_k(
                    tf.argmax(labels_plant, 1),
                    logits_plant,
                    k=5,
                    weights=mask_plant)

            metrics = {
                'logits_animal': logits_animal,
                'logits_plant': logits_plant,
                'labels_animal': labels['animal_label'],
                'labels_plant': labels['plant_label'],
                'mask_animal': labels['animal_mask'],
                'mask_plant': labels['plant_mask'],
            }

            return tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode, loss=loss, eval_metrics=(metric_fn, metrics),
                scaffold_fn=None)
        else:
            print('Invalid mode.')

    return model_fn


def build_model_fn(model, num_classes, num_train_examples):
    """Build model function."""
    def model_fn(features, labels, mode, params=None):
        """Build model and optimizer."""
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Check training mode.
        if FLAGS.train_mode == 'pretrain':
          num_transforms = 2
          if FLAGS.fine_tune_after_block > -1:
            raise ValueError('Does not support layer freezing during pretraining,'
                             'should set fine_tune_after_block<=-1 for safety.')
        elif FLAGS.train_mode == 'finetune':
          num_transforms = 1
        else:
          raise ValueError('Unknown train_mode {}'.format(FLAGS.train_mode))

        # Split channels, and optionally apply extra batched augmentation.
        features_list = tf.split(features, num_or_size_splits=num_transforms,
                                 axis=-1)
        if FLAGS.use_blur and is_training and FLAGS.train_mode == 'pretrain':
            features_list = data_util.batch_random_blur(
                features_list, FLAGS.image_size, FLAGS.image_size)
        features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)

        # Base network forward pass.
        with tf.variable_scope('base_model'):
            if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block >= 4:
                # Finetune just supervised (linear) head will not update BN stats.
                model_train_mode = False
            else:
                # Pretrain or finetuen anything else will update BN stats.
                model_train_mode = is_training
            hiddens = model(features, is_training=model_train_mode)

        # Add head and loss.
        if FLAGS.train_mode == 'pretrain':
            tpu_context = params['context'] if 'context' in params else None
            hiddens_proj = model_util.projection_head(hiddens, is_training)
            contrast_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
              hiddens_proj,
              hidden_norm=FLAGS.hidden_norm,
              temperature=FLAGS.temperature,
              tpu_context=tpu_context if is_training else None)
            logits_sup = tf.zeros([params['batch_size'], num_classes])
        else:
            contrast_loss = tf.zeros([])
            logits_con = tf.zeros([params['batch_size'], 10])
            labels_con = tf.zeros([params['batch_size'], 10])
            logits_sup = model_util.supervised_head(
              hiddens, num_classes, is_training)
            obj_lib.add_supervised_loss(
              labels=labels['labels'],
              logits=logits_sup,
              weights=labels['mask'])

        # Add weight decay to loss, for non-LARS optimizers.
        model_util.add_weight_decay(adjust_per_optimizer=True)
        loss = tf.losses.get_total_loss()

        if FLAGS.train_mode == 'pretrain':
            variables_to_train = tf.trainable_variables()
        else:
            collection_prefix = 'trainable_variables_inblock_'
            variables_to_train = []
            for j in range(FLAGS.fine_tune_after_block + 1, 6):
                variables_to_train += tf.get_collection(collection_prefix + str(j))
            assert variables_to_train, 'variables_to_train shouldn\'t be empty!'

        tf.logging.info('===============Variables to train (begin)===============')
        tf.logging.info(variables_to_train)
        tf.logging.info('================Variables to train (end)================')

        learning_rate = model_util.learning_rate_schedule(
            FLAGS.learning_rate, num_train_examples)

        if is_training:
            if FLAGS.train_summary_steps > 0:
                # Compute stats for the summary.
                prob_con = tf.nn.softmax(logits_con)
                entropy_con = - tf.reduce_mean(
                    tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))

                summary_writer = tf2.summary.create_file_writer(FLAGS.model_dir)
                # TODO(iamtingchen): remove this control_dependencies in the future.
                with tf.control_dependencies([summary_writer.init()]):
                    with summary_writer.as_default():
                        should_record = tf.math.equal(
                            tf.math.floormod(tf.train.get_global_step(),
                                             FLAGS.train_summary_steps), 0)
                        with tf2.summary.record_if(should_record):
                            contrast_acc = tf.equal(
                              tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
                            contrast_acc = tf.reduce_mean(tf.cast(contrast_acc, tf.float32))
                            label_acc = tf.equal(
                              tf.argmax(labels['labels'], 1), tf.argmax(logits_sup, axis=1))
                            label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
                            tf2.summary.scalar(
                              'train_contrast_loss',
                              contrast_loss,
                              step=tf.train.get_global_step())
                            tf2.summary.scalar(
                              'train_contrast_acc',
                              contrast_acc,
                              step=tf.train.get_global_step())
                            tf2.summary.scalar(
                              'train_label_accuracy',
                              label_acc,
                              step=tf.train.get_global_step())
                            tf2.summary.scalar(
                              'contrast_entropy',
                              entropy_con,
                              step=tf.train.get_global_step())
                            tf2.summary.scalar(
                              'learning_rate', learning_rate,
                              step=tf.train.get_global_step())
                            tf2.summary.scalar(
                              'input_mean',
                              tf.reduce_mean(features),
                              step=tf.train.get_global_step())
                            tf2.summary.scalar(
                              'input_max',
                              tf.reduce_max(features),
                              step=tf.train.get_global_step())
                            tf2.summary.scalar(
                              'input_min',
                              tf.reduce_min(features),
                              step=tf.train.get_global_step())
                            tf2.summary.scalar(
                              'num_labels',
                              tf.reduce_mean(tf.reduce_sum(labels['labels'], -1)),
                              step=tf.train.get_global_step())

            if FLAGS.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate, FLAGS.momentum, use_nesterov=True)
            elif FLAGS.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate)
            elif FLAGS.optimizer == 'lars':
                optimizer = LARSOptimizer(
                    learning_rate,
                    momentum=FLAGS.momentum,
                    weight_decay=FLAGS.weight_decay,
                    exclude_from_weight_decay=['batch_normalization', 'bias'])
            else:
                raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))

            if FLAGS.use_tpu:
                optimizer = tf.tpu.CrossShardOptimizer(optimizer)

            control_deps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if FLAGS.train_summary_steps > 0:
                control_deps.extend(tf.summary.all_v2_summary_ops())
            with tf.control_dependencies(control_deps):
                train_op = optimizer.minimize(
                    loss, global_step=tf.train.get_or_create_global_step(),
                    var_list=variables_to_train)

            if FLAGS.checkpoint:
                def scaffold_fn():
                    """Scaffold function to restore non-logits vars from checkpoint."""
                    for v in tf.global_variables(FLAGS.variable_schema):
                        print(v.op.name)
                    tf.train.init_from_checkpoint(
                        FLAGS.checkpoint,
                        {v.op.name: v.op.name
                            for v in tf.global_variables(FLAGS.variable_schema)})

                    if FLAGS.zero_init_logits_layer:
                        # Init op that initializes output layer parameters to zeros.
                        output_layer_parameters = [
                            var for var in tf.trainable_variables() if var.name.startswith(
                                'head_supervised')]
                        tf.logging.info('Initializing output layer parameters %s to zero',
                                        [x.op.name for x in output_layer_parameters])
                        with tf.control_dependencies([tf.global_variables_initializer()]):
                            init_op = tf.group(
                                [tf.assign(x, tf.zeros_like(x))
                                    for x in output_layer_parameters])
                        return tf.train.Scaffold(init_op=init_op)
                    else:
                        return tf.train.Scaffold()
            else:
                scaffold_fn = None

            return tf.estimator.tpu.TPUEstimatorSpec(
              mode=mode, train_op=train_op, loss=loss, scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            _, top_5 = tf.nn.top_k(logits_sup, k=5)
            predictions = {
                'label': tf.argmax(labels['labels'], 1),
                'top_5': top_5,
            }
            return tf.estimator.tpu.TPUEstimatorSpec(
              mode=mode,
              predictions=predictions)
        else:
            def metric_fn(logits_sup, labels_sup, logits_con, labels_con, mask,
                          **kws):
                """Inner metric function."""
                metrics = {k: tf.metrics.mean(v, weights=mask)
                           for k, v in kws.items()}
                metrics['label_top_1_accuracy'] = tf.metrics.accuracy(
                    tf.argmax(labels_sup, 1), tf.argmax(logits_sup, axis=1),
                    weights=mask)
                metrics['label_top_5_accuracy'] = tf.metrics.recall_at_k(
                    tf.argmax(labels_sup, 1), logits_sup, k=5, weights=mask)
                metrics['contrastive_top_1_accuracy'] = tf.metrics.accuracy(
                    tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1),
                    weights=mask)
                metrics['contrastive_top_5_accuracy'] = tf.metrics.recall_at_k(
                    tf.argmax(labels_con, 1), logits_con, k=5, weights=mask)

                metrics['mean_class_accuracy'] = tf.metrics.mean_per_class_accuracy(
                        tf.argmax(labels_sup, 1),
                        tf.argmax(logits_sup, axis=1), num_classes,
                        weights=mask, name='mca')

                running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                                 scope="mca")
                metrics['mean_class_accuracy_total'] = running_vars[0]
                metrics['mean_class_accuracy_count'] = running_vars[1]

                return metrics

            metrics = {
              'logits_sup': logits_sup,
              'labels_sup': labels['labels'],
              'logits_con': logits_con,
              'labels_con': labels_con,
              'mask': labels['mask'],
              'contrast_loss': tf.fill((params['batch_size'],), contrast_loss),
              'regularization_loss': tf.fill((params['batch_size'],),
                                             tf.losses.get_regularization_loss()),
            }

            return tf.estimator.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=loss,
              eval_metrics=(metric_fn, metrics),
              scaffold_fn=None)

    return model_fn
