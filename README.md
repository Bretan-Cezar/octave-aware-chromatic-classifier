# Electric Guitar DI Signal Octave-Aware Chromatic Note Classifier

## General

This repository contains implementation and documentation for a neural network-based
musical note classifier. Takes electric guitar DI signals, and outputs labels corresponding
to detected notes between A0 - G#6.

## model.py

Currently, the model itself is MLP-based and works only for single notes. Outputs a probability vector 
for the 72 pitch labels.

## feature_extractor.py

Feature extractor settings are configurable via **data_config.json**, uses STFT to obtain log magnitude spectrums from audio of the plucked notes, 
pairs each of them with the corresponding pitches and dumps them as h5py databases.

## extract.py

Script for loading extractor settings and begin the extraction process.

## dataset.py

Dataset class derived from the PyTorch IterableDataset. Loads the spectrum - pitch label pairs from h5py files.
Meant to be used for data loader instantiation, pairs are shuffled whenever the iterator is obtained.

## train.py

Contains the function for the model training loop, and the script code for loading settings for training from **train_config.json**.

## test.py

Contains the function for running inference on test data, and the script code for running it separately from training.

## test_realtime.py

Script for connecting to an ASIO audio input, starting a Python sounddevice stream, and consuming the stream in a buffered manner.
Settings are loaded from **realtime_config.json**, where a spectral visualizer or the model inference can be enabled.
The spectral visualizer simply outputs the log magnitude spectrum for the current audio buffer state via a matplotlib figure.
In model inference mode, the argmax of the probability vector output is taken, converted to a string describing the note, 
and shown in the console whenever a new note is detected.