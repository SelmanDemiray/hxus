# Text Chat Neural Network

This is a Python application that implements a text chat neural network using an encoder-decoder architecture.

## Overview

The application contains:
- A data processor for text preprocessing
- A custom encoder-decoder neural network implementation
- Training functionality with forward and backward pass
- Model saving and loading in multiple formats
- A chat interface for testing models

## Requirements

- Python 3.6+
- NumPy
- h5py
- pickle

## Usage

### Training the Model

```bash
python main.py --mode train --epochs 50 --batch_size 32 --learning_rate 0.01
```

By default, the application uses sample conversation data. You can adjust hyperparameters like epochs, batch size, learning rate, embedding dimension, and hidden dimension.

### Chat with a Trained Model

```bash
python main.py --mode chat --model_path models/model.pkl --processor_path models/processor.pkl
```

This starts an interactive chat session with a trained model.

### Test a Trained Model

```bash
python main.py --mode test --model_path models/model.pkl --processor_path models/processor.pkl
```

This tests the model on a set of predefined questions.

## Model Architecture

The encoder-decoder architecture consists of:
1. Encoder: Processes the input sequence and produces a context vector
2. Decoder: Generates the output sequence based on the context vector

## File Formats

The model is saved in multiple formats:
- `.pkl` - Pickle format
- `.json` - JSON format
- `.h5` - HDF5 format
- `.npz` - NumPy compressed format

## Implementation Details

This implementation follows a homemade approach inspired by the MATLAB code provided. It includes:
- Custom forward pass implementation
- Custom backward pass with gradient computation
- Weight updates with learning rate and momentum
- Sigmoid and softmax activation functions
