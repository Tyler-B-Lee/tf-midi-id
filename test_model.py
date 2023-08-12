# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Evaluates the midi network on a single midi file."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
from matplotlib import pyplot as plt

# from six.moves import xrange  # pylint: disable=redefined-builtin
# import tensorflow as tf
import tensorflow.compat.v1 as tf

import input_data
import midi

FLAGS = {
  'learning_rate': 0.01,
  'max_steps': 2000, 
  'hidden1': 1024, 
  'hidden2': 16, 
  'batch_size': 1, 
  'train_dir': 'midis'
}

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    midi_data_placeholder: MIDI data placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  midi_data_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         input_data.INPUT_SIZE))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return midi_data_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS['batch_size'])
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            midi_data_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    midi_data_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of midi data and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  steps_per_epoch = data_set.num_examples
  ret = {i:[] for i in range(midi.NUM_CLASSES)}
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               midi_data_placeholder,
                               labels_placeholder)
    output = sess.run(eval_correct,feed_dict=feed_dict)[0]
    for i in range(midi.NUM_CLASSES):
      ret[i].append(output[i])
  return ret

def visualize_results(output_dict:dict):
  fig, axs = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(12,6))
  x = numpy.arange(len(output_dict[0]))
  for i,ax in enumerate(axs):
    ax.set_title(input_data.INDEX_TO_COMP[i])
    ax.plot(x,output_dict[i])
    ax.plot(x,[0]*len(x))
  
  fig.suptitle('Predicted Style Outputs')
  plt.show()


def run_test():
  """Train midi data for a number of steps."""
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    midi_data_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS['batch_size'])

    # Build a Graph that computes predictions from the inference model.
    logits = midi.inference(midi_data_placeholder,
                             FLAGS['hidden1'],
                             FLAGS['hidden2'])

    # Add to the Graph the Ops for loss calculation.
    loss = midi.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = midi.training(loss, FLAGS['learning_rate'])

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = midi.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    # init = tf.initialize_all_variables()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    print("Restoring model...")
    saver.restore(sess, "midis-1999")

    # And then after everything is built, start testing indefinitely.
    while True:
      input_dir = input("Enter directory to use (no quotes): ")
      target_name = "abcdtest"
      while True:
        target_name = input("Enter midi file name: ")
        if target_name == "0":
          break

        data_set = None
        # for (dirpath, dirnames, filenames) in os.walk(input_dir):
        #   for filename in [f for f in filenames if f.endswith('.csv')]:
        #     if filename == target_name:
        #       # build temporary dataset to use
        #       midi_data = input_data.extract_midi_data_no_dup(os.path.join(dirpath,filename))
        #       labels = numpy.array([0]*midi_data.shape[0])
        #       data_set = input_data.DataSet(midi_data,labels)

        for (dirpath, dirnames, filenames) in os.walk(input_dir):
          for filename in [f for f in filenames if f.endswith('.mid')]:
            if filename == target_name:
              # translate midi file to csv
              target_filename = rf"{dirpath}\{filename}"
              os.system(f'midicsv "{target_filename}" "translated.csv"')
              # build temporary dataset to use
              midi_data = input_data.extract_midi_data_no_dup('translated.csv')
              labels = numpy.array([0]*midi_data.shape[0])
              data_set = input_data.DataSet(midi_data,labels)

        if data_set is None:
          print(f"\tMidi file {target_name} not found / dataset unsuccessfully built.")
          continue

        # Evaluate against the file's data.
        print('Evaluating chosen file:')
        results = do_eval(sess,
                logits,
                midi_data_placeholder,
                labels_placeholder,
                data_set)
        # visualize results
        visualize_results(results)


def main(_):
  run_test()


if __name__ == '__main__':
  tf.app.run()
