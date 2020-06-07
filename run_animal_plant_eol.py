from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import csv
from absl import app
from absl import flags

import resnet
import data_eol as data_lib
import model as model_lib
import model_util as model_util

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float(
    'weight_decay', 1e-6,
    'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'train_batch_size', 512,
    'Batch size for training.')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_steps', 0,
    'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'train_summary_steps', 100,
    'Steps before saving training summaries. If 0, will not save.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string(
    'eval_split', 'val',
    'Split for evaluation.')

flags.DEFINE_string(
    'dataset', 'eol2020',
    'Name of a dataset.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval', 'predict'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for continued training or fine-tuning.')

flags.DEFINE_string(
    'variable_schema', '?!global_step',
    'This defines whether some variable from the checkpoint should be loaded.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning'
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linera head.')

flags.DEFINE_string(
    'master', None,
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_bool(
    'use_tpu', True,
    'Whether to run on TPU.')

tf.flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

tf.flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

tf.flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_enum(
    'optimizer', 'lars', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 2,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_enum(
    'head_proj_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'head_proj_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_nlh_layers', 1,
    'Number of non-linear head layers.')

flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 50,
    'Depth of ResNet.')

flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')

flags.DEFINE_string(
    'gcs_bucket_name', None,
    'GCS Bucket Name where tfrecords reside')

flags.DEFINE_string(
    'train_path', None,
    'Path to train tfrecords')

flags.DEFINE_string(
    'val_path', None,
    'Path to val tfrecords')

flags.DEFINE_string(
    'labels_path', None,
    'Path to file with combined animal_plant_labels')

flags.DEFINE_string(
    'animal_labels_path', None,
    'Path to file with animal labels')

flags.DEFINE_string(
    'plant_labels_path', None,
    'Path to file with plant labels')

flags.DEFINE_string(
    'dset_stats_path', None,
    'path to JSON containing dataset stats')

flags.DEFINE_string(
    'predict_output_path', None,
    'path to store prediction output')


def build_hub_module(model, animal_num_classes, plant_num_classes,
                     global_step, checkpoint_path):
    """Create TF-Hub module."""

    tags_and_args = [
      # The default graph is built with batch_norm, dropout etc. in inference
      # mode. This graph version is good for inference, not training.
      ([], {'is_training': False}),
      # A separate "train" graph builds batch_norm, dropout etc. in training
      # mode.
      (['train'], {'is_training': True}),
    ]

    def module_fn(is_training):
        """Function that builds TF-Hub module."""
        endpoints = {}
        inputs = tf.placeholder(
            tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
        with tf.variable_scope('base_model', reuse=tf.AUTO_REUSE):
            hiddens = model(inputs, is_training)
            for v in ['initial_conv', 'initial_max_pool', 'block_group1',
                      'block_group2', 'block_group3', 'block_group4',
                      'final_avg_pool']:
                endpoints[v] = tf.get_default_graph().get_tensor_by_name(
                    'base_model/{}:0'.format(v))
        if FLAGS.train_mode == 'pretrain':
            print('Pretrain mode not available.')
        else:
            logits_animal = model_util.supervised_head(hiddens,
                                                       animal_num_classes,
                                                       is_training)
            logits_plant = model_util.supervised_head(hiddens,
                                                      plant_num_classes,
                                                      is_training)
            endpoints['logits_animal'] = logits_animal
            endpoints['logits_plant'] = logits_plant
            hub.add_signature(inputs=dict(images=inputs),
                              outputs=dict(endpoints, default=hiddens))

    # Drop the non-supported non-standard graph collection.
    drop_collections = ['trainable_variables_inblock_%d' %d for d in range(6)]
    spec = hub.create_module_spec(module_fn, tags_and_args, drop_collections)
    hub_export_dir = os.path.join(FLAGS.model_dir, 'hub')
    checkpoint_export_dir = os.path.join(hub_export_dir, str(global_step))
    if tf.io.gfile.exists(checkpoint_export_dir):
        # Do not save if checkpoint already saved.
        tf.io.gfile.rmtree(checkpoint_export_dir)
        spec.export(
          checkpoint_export_dir,
          checkpoint_path=checkpoint_path,
          name_transform_fn=None)

    if FLAGS.keep_hub_module_max > 0:
        # Delete old exported Hub modules.
        exported_steps = []
        for subdir in tf.io.gfile.listdir(hub_export_dir):
            if not subdir.isdigit():
                continue
            exported_steps.append(int(subdir))
        exported_steps.sort()
        for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
            tf.io.gfile.rmtree(os.path.join(hub_export_dir,
                               str(step_to_delete)))


def perform_evaluation(estimator, input_fn, eval_steps, model,
                       animal_num_classes, plant_num_classes,
                       checkpoint_path=None):
    if not checkpoint_path:
        checkpoint_path = estimator.latest_checkpoint()
    result = estimator.evaluate(
        input_fn, eval_steps, checkpoint_path=checkpoint_path,
        name=FLAGS.eval_name)

    # Record results as JSON.
    result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    result_json_path = os.path.join(
        FLAGS.model_dir, 'result_%d.json'%result['global_step'])
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')
    with tf.io.gfile.GFile(flag_json_path, 'w') as f:
        json.dump(FLAGS.flag_values_dict(), f)

    # Save Hub module.
    build_hub_module(model, animal_num_classes, plant_num_classes,
                     global_step=result['global_step'],
                     checkpoint_path=checkpoint_path)

    return result


def perform_predict(estimator, input_fn, checkpoint_path=None):
    """
    Perform prediction.
    Args:
    estimator: TPUEstimator instance.
    input_fn: Input function for estimator.
    checkpoint_path: Path of checkpoint to evaluate.

    Returns:
    None
    """
    if not checkpoint_path:
        checkpoint_path = estimator.latest_checkpoint()
    est_output = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

    with open(FLAGS.predict_output_path, 'a+') as f:
        writer = csv.writer(f)
        for item in est_output:
            writer.writerow(
                [item['label']] + [item['animal_label']] +
                [item['plant_label']] + list(item['animal_top_5']) +
                list(item['plant_top_5']))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if FLAGS.train_summary_steps > 0:
        tf.config.set_soft_device_placement(True)

    # Dataset statistics
    with open(FLAGS.dset_stats_path, 'r') as f:
        dset_stats = json.load(f)

    num_train_examples = dset_stats['num_train']
    num_eval_examples = dset_stats['num_val']
    tot_num_classes = dset_stats['num_classes']
    animal_num_classes = dset_stats['animal_classes']
    plant_num_classes = dset_stats['plant_classes']

    assert tot_num_classes == animal_num_classes + plant_num_classes

    train_steps = model_util.get_train_steps(num_train_examples)
    eval_steps = int(math.ceil(num_eval_examples / FLAGS.eval_batch_size))
    epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

    resnet.BATCH_NORM_DECAY = FLAGS.batch_norm_decay
    model = resnet.resnet_v1(
        resnet_depth=FLAGS.resnet_depth,
        width_multiplier=FLAGS.width_multiplier,
        cifar_stem=FLAGS.image_size <= 32)

    checkpoint_steps = (FLAGS.checkpoint_steps or (
                        FLAGS.checkpoint_epochs * epoch_steps))

    cluster = None
    if FLAGS.use_tpu and FLAGS.master is None:
        if FLAGS.tpu_name:
            cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        else:
            cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(cluster)
            tf.tpu.experimental.initialize_tpu_system(cluster)

    default_eval_mode = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V1
    sliced_eval_mode = tf.estimator.tpu.InputPipelineConfig.SLICED

    run_config = tf.estimator.tpu.RunConfig(
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=checkpoint_steps,
            eval_training_input_configuration=sliced_eval_mode
            if FLAGS.use_tpu else default_eval_mode),
        model_dir=FLAGS.model_dir,
        save_summary_steps=checkpoint_steps,
        save_checkpoints_steps=checkpoint_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        master=FLAGS.master,
        cluster=cluster)

    estimator = tf.estimator.tpu.TPUEstimator(
        model_lib.build_multi_kingdom_model_fn(model, animal_num_classes,
                                               plant_num_classes,
                                               num_train_examples),
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.eval_batch_size,
        use_tpu=FLAGS.use_tpu)

    if FLAGS.mode == 'eval':
        perform_evaluation(
            estimator=estimator,
            input_fn=data_lib.build_multi_kingdom_input_fn(
                tot_num_classes, animal_num_classes,
                plant_num_classes, False),
            eval_steps=eval_steps,
            model=model,
            animal_num_classes=animal_num_classes,
            plant_num_classes=plant_num_classes)
    elif FLAGS.mode == 'predict':
        perform_predict(estimator=estimator,
                        input_fn=data_lib.build_multi_kingdom_input_fn(
                            tot_num_classes, animal_num_classes,
                            plant_num_classes, False)
                        )
    else:
        estimator.train(data_lib.build_multi_kingdom_input_fn(
            tot_num_classes, animal_num_classes, plant_num_classes, True),
            max_steps=train_steps)
        if FLAGS.mode == 'train_then_eval':
            perform_evaluation(
                estimator=estimator,
                input_fn=data_lib.build_multi_kingdom_input_fn(
                    tot_num_classes, animal_num_classes,
                    plant_num_classes, False),
                eval_steps=eval_steps,
                model=model,
                animal_num_classes=animal_num_classes,
                plant_num_classes=plant_num_classes)


if __name__ == '__main__':
    # Disable eager mode when running with TF2.
    tf.disable_eager_execution()
    app.run(main)
