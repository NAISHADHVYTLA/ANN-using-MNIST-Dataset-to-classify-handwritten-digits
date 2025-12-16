# ANN-using-MNIST-Dataset-to-classify-handwritten-digits
This experiment built and trained a neural network using the MNIST dataset to classify handwritten digits. The model achieved 97.38% accuracy on the test set after 10 epochs of training.
```markdown
# MNIST Digit Classification using a Simple Neural Network

## Project Description

This experiment demonstrates the implementation and training of a basic feedforward neural network to classify handwritten digits from the MNIST dataset. The goal is to build a model that can accurately identify digits (0-9) from grayscale images.

## Prerequisites

To run this experiment, you should have:

*   **Python 3.x** installed.
*   **Basic understanding of Machine Learning**: Concepts like supervised learning, classification, neural networks, loss functions, and accuracy.
*   **Familiarity with TensorFlow/Keras**: Knowledge of how to define, compile, and train neural networks using these libraries.
*   **Jupyter/Colab environment**: Experience running code in interactive notebooks.

## Installation

This experiment primarily uses `tensorflow` and `matplotlib`. You can install them using pip:

```bash
pip install tensorflow matplotlib numpy
```

## Model Architecture

The neural network architecture consists of the following layers:

*   **Input Layer**: `Flatten` layer to convert the 28x28 pixel images into a 784-neuron vector.
*   **Hidden Layers**: Two `Dense` layers, each with 128 neurons and using the `ReLU` (Rectified Linear Unit) activation function.
*   **Output Layer**: A `Dense` layer with 10 neurons (one for each digit class) and a `Softmax` activation function to output probability distributions over the classes.

**Total Trainable Parameters**: 118,282

## Training Details

*   **Optimizer**: Adam
*   **Loss Function**: Sparse Categorical Crossentropy
*   **Epochs**: 10
*   **Batch Size**: 100
*   **Validation Split**: 20% of the training data was used for validation during training.

## Dataset

The experiment uses the **MNIST dataset**, which contains:

*   60,000 training images (28x28 grayscale)
*   10,000 testing images (28x28 grayscale)

Data was normalized using `tf.keras.utils.normalize`.

## Results

After 10 epochs of training, the model achieved the following performance on the test set:

*   **Test Accuracy**: 97.38%
*   **Test Loss**: 0.0916

Training performance:
*   **Training Accuracy**: 99.54%
*   **Training Loss**: 0.018%


## Usage

To run this experiment:

1.  Clone this repository (if hosted on GitHub).
2.  Open the `.ipynb` notebook in Google Colab or a Jupyter environment.
3.  Run all cells sequentially to load the data, build the model, train it, and evaluate its performance.

```python
# Example of how to load and predict a single image
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_test and model are already defined and trained
# from the notebook content.

# Make a prediction for a sample image (e.g., the 10th image in the test set)
prediction = model.predict([X_test])

# Get the predicted digit and probabilities
predicted_digit = np.argmax(prediction[10])
probabilities = prediction[10]

print(f"Probabilities: {probabilities}")
print(f"Predicted Digit: {predicted_digit}")

# Visualize the image
plt.imshow(X_test[10])
plt.title(f"True Label: {y_test[10]}, Predicted: {predicted_digit}")
plt.show()
```

