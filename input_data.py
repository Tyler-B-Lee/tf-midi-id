"""Functions for downloading and reading MNIST data."""
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy
# import tensorflow as tf
# import tensorflow.compat.v1 as tf

SAMPLE_DURATION_MILLIS = 10000
SAMPLE_WINDOW_MILLIS = 100
MAX_NOTE_VALUE = 100
INPUT_SIZE = int(SAMPLE_DURATION_MILLIS / SAMPLE_WINDOW_MILLIS * MAX_NOTE_VALUE)
SAMPLE_SIZE = int(SAMPLE_DURATION_MILLIS / SAMPLE_WINDOW_MILLIS)

COMPOSERS = {'bach': 0, 'mozart': 1, 'beethoven': 2, 'chopin': 3, 'rach': 4}
INDEX_TO_COMP = {ind:comp for comp,ind in COMPOSERS.items()}

def convert_midi_to_ms(filename):
  """
  Attempts to convert a given midi file to an array of midi_data:
  - [(start time of note in milliseconds), note number - 21 (adjusted to fit within 0 to 100), ...]

  This is what is done in the original extract_midi_data function, but it assumes that the song is
  saved with each midi data event equalling exactly 1 millisecond. This is not accurate for songs
  that have a different tempo. In this function, a data structure is generated in the same format as
  the original extraction function, but with the times adjusted.
  """
  tempo_changes = []
  note_data = []
  with open(filename, encoding='ISO-8859-1') as f:
    for line in f:
      fields = line.split(', ')
      if fields[2] == 'Note_on_c':
        # time and note value
        if fields[5] == "0":
          continue
        note_data.append([int(fields[1]), int(fields[4]) - 21])
      elif fields[2] == 'Tempo':
        tempo_changes.append([int(fields[1]), int(fields[3])])
      elif fields[2] == 'Header':
        if fields[3] == "1":
          conversion_constant = 500 / int(fields[5])
        else:
          print(f"\t\tINFO: Type 0? MIDI found with value {fields[5]} in {filename}")
          conversion_constant = 9.6
        
  note_data = sorted(note_data, key=lambda event: event[0])
  tempo_changes = sorted(tempo_changes, key=lambda event: event[0])

  print(f"\tNotes found: {len(note_data)}")
  print(f"\tTempo Marks found: {len(tempo_changes)}")

  note_data_arr = numpy.array(note_data, dtype=numpy.int32)
  if len(tempo_changes) == 0 or tempo_changes[0][0] != 0:
    # print("\tInserting default tempo at time 0...")
    tempo_changes.insert(0, [0,500_000])

  # create scaled sections after each tempo change
  scaled_sections = []
  for tempo_change_index,tc in enumerate(tempo_changes):
    # print(f"Tempo change {tempo_change_index}: Change to {tc[1]} at tick {tc[0]}")
    start_tempo_tick = tc[0] # this is the tick in the unconverted song we start at
    # we then find the index of the first note-on event to convert in the note data array
    if tempo_change_index == 0:
      start_nd_row = 0
    else:
      start_nd_row = max(numpy.argmax(note_data_arr[:,0] > start_tempo_tick), 0)
    conversion_multiplier = tc[1] / 500_000 * conversion_constant
    # find which notes we need to rescale
    if tempo_change_index + 1 == len(tempo_changes):
      # if are doing the last tempo change, take up to the last note
      end_nd_row = len(note_data_arr)
    else:
      # otherwise, cut up until the next tempo change
      end_tempo_tick = tempo_changes[tempo_change_index + 1][0]
      end_nd_row = max(numpy.argmax(note_data_arr[:,0] > end_tempo_tick), 0)
    
    # print(f"\tStart / End Rows: {start_nd_row} - {end_nd_row}")
    slice_to_convert = note_data_arr[start_nd_row:end_nd_row].copy()
    slice_to_convert[:,0] -= start_tempo_tick
    slice_to_convert[:,0] = (slice_to_convert[:,0] * conversion_multiplier).astype(numpy.int32)

    scaled_sections.append(slice_to_convert)
  
  # combine all of the sections
  out = numpy.zeros((0,2),dtype=numpy.int32)
  a = 0
  for section in scaled_sections:
    section[:,0] += a
    if len(section) > 0:
      a = section[-1][0]
    out = numpy.vstack((out,section))
  
  return out  

def extract_midi_data(filename):
  midi_data = []
  with open(filename) as f:
    for line in f:
      fields = line.split(', ')
      if len(fields) < 5:
        continue
      if fields[2] != 'Note_on_c':
        continue
      # time and note value
      midi_data.append([float(fields[1]), int(fields[4]) - 21])
  midi_data = sorted(midi_data, key=lambda event: event[0])
  data = None
  # artifically increase sample data by shifting the data into different windows
  for x in range(10):
    while midi_data[0][0] < x:
      midi_data.pop(0)
    shiftdata = numpy.zeros((int(math.ceil(midi_data[-1][0]/SAMPLE_WINDOW_MILLIS))+1, MAX_NOTE_VALUE), dtype=int)
    for event in midi_data:
      curtime = event[0]
      note = event[1]
      shiftdata[int(math.floor(curtime/SAMPLE_WINDOW_MILLIS)), note] += 1
    if data is None:
      data = shiftdata
    else:
      data = numpy.vstack((data, shiftdata))

  # make sure we have full 10-second samples
  sample_data_length = int(SAMPLE_DURATION_MILLIS/SAMPLE_WINDOW_MILLIS)
  windows = int(data.shape[0]/sample_data_length)*sample_data_length
  data = numpy.resize(data, (windows, data.shape[1]))

  print('Read %d windows from %s' % (windows, filename))
  
  return data.reshape((int(data.shape[0]/sample_data_length), int(data.shape[1]*sample_data_length)))

def extract_midi_data_v2(filename):
  data = convert_midi_to_ms(filename)
  # # artifically increase sample data by shifting the data into different windows
  # for x in range(SAMPLE_WINDOW_MILLIS):
  #   while midi_data[0][0] < x:
  #     # midi_data.pop(0)
  #     midi_data = numpy.delete(midi_data, 0, 0)
  #   shiftdata = numpy.zeros((int(math.ceil(midi_data[-1][0]/SAMPLE_WINDOW_MILLIS))+1, MAX_NOTE_VALUE), dtype=int)
  #   for event in midi_data:
  #     curtime = event[0]
  #     note = event[1]
  #     shiftdata[int(math.floor(curtime/SAMPLE_WINDOW_MILLIS)), note] += 1
  #   if data is None:
  #     data = shiftdata
  #   else:
  #     data = numpy.vstack((data, shiftdata))

  # make sure we have full 10-second samples
  sample_data_length = int(SAMPLE_DURATION_MILLIS/SAMPLE_WINDOW_MILLIS)
  windows = int(data.shape[0]/sample_data_length)*sample_data_length
  data = numpy.resize(data, (windows, data.shape[1]))

  print('Read %d windows from %s' % (windows, filename))
  
  return data.reshape((int(data.shape[0]/sample_data_length), int(data.shape[1]*sample_data_length)))

def extract_midi_data_no_dup(filename):
  # midi_data = []
  # with open(filename) as f:
  #   for line in f:
  #     fields = line.split(', ')
  #     if len(fields) < 5:
  #       continue
  #     if fields[2] != 'Note_on_c':
  #       continue
  #     if fields[5] == 0:
  #       continue
  #     # time and note value
  #     midi_data.append([float(fields[1]), int(fields[4]) - 21])
  # midi_data = sorted(midi_data, key=lambda event: event[0])
  midi_data = convert_midi_to_ms(filename)

  # read the file once
  data = numpy.zeros((int(math.ceil(midi_data[-1][0]/SAMPLE_WINDOW_MILLIS))+1, MAX_NOTE_VALUE), dtype=int)
  for event in midi_data:
    curtime = event[0]
    note = event[1]
    data[int(math.floor(curtime/SAMPLE_WINDOW_MILLIS)), note] += 1

  # make sure we have full 10-second samples
  sample_data_length = int(SAMPLE_DURATION_MILLIS/SAMPLE_WINDOW_MILLIS)
  windows = int(data.shape[0]/sample_data_length)*sample_data_length
  pad_length = sample_data_length + windows - data.shape[0]
  data = numpy.pad(data,((0,pad_length),(0,0)),'constant',constant_values=0)

  print('Read %d windows from %s' % (data.shape[0], filename))
  
  return data.reshape((int(data.shape[0]/sample_data_length), int(data.shape[1]*sample_data_length)))

def extract_data(dirname):
  print('Extracting', dirname)
  labels = numpy.array([])
  data = None
  for (dirpath, dirnames, filenames) in os.walk(dirname):
    for filename in [f for f in filenames if f.endswith('.csv')]:
      labelval = COMPOSERS[os.path.split(dirpath)[-1]]
      #label = numpy.zeros(len(COMPOSERS), dtype=numpy.uint8)
      #label[labelval] = 1

      midi_data = extract_midi_data_v2(os.path.join(dirpath, filename))
      # midi_data = extract_midi_data(os.path.join(dirpath, filename))

      if data is None:
        data = midi_data
      else:
        data = numpy.vstack((data, midi_data))
      labels = numpy.append(labels, [labelval]*midi_data.shape[0])
      #for x in range(0, midi_data.shape[0]):
      #  if labels is None:
      #    labels = label
      #  else:
      #    labels = numpy.vstack((labels, label))

  return DataSet(data, labels)

class DataSet(object):

  def __init__(self, midi_data, labels):
    assert midi_data.shape[0] == labels.shape[0], ('midi_data.shape: %s labels.shape: %s' % (midi_data.shape, labels.shape))
    self._num_examples = midi_data.shape[0]
    self._midi_data = midi_data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def midi_data(self):
    return self._midi_data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._midi_data = self._midi_data[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._midi_data[start:end], self._labels[start:end]


def read_data_sets(train_dir):
  class DataSets(object):
    pass
  data_sets = DataSets()

  data_sets.train = extract_data(os.path.join(train_dir, 'train'))
  data_sets.validation = extract_data(os.path.join(train_dir, 'validation'))
  data_sets.test = extract_data(os.path.join(train_dir, 'test'))

  return data_sets


def load_midis():
    return read_data_sets("midis")

