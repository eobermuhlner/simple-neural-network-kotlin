# Simple Neural Network in Kotlin

This project provides a straightforward implementation of a neural network in Kotlin, designed for educational purposes and ease of understanding. The neural network is implemented without the use of external machine learning libraries, making it an excellent resource for those looking to understand the fundamentals of neural networks and their implementation details.

## Features
- Fully Implemented in Kotlin: Utilizes Kotlin's concise syntax and features for clear and readable code.
- No External Dependencies: Built from the ground up without relying on external machine learning libraries.
- Customizable Neural Network Architecture: Easily configure the number of layers and neurons to fit various problem requirements. 
- Simple API: Designed to be accessible to beginners, with straightforward methods for training and using the network.

## Getting Started

To get started with this neural network, clone the repository and import it into your preferred IDE that supports Kotlin projects. 
Ensure you have Kotlin set up in your development environment.

## Example: MnistMain

The `MnistMain` class demonstrates a practical application of the neural network using the MNIST dataset, a large database of handwritten digits commonly used for training and testing in the field of machine learning.

### Setup and Execution

To run the MnistMain example:

- Ensure the MNIST dataset is available in the expected format and location.
  - Create the directory `data/mnist` in the root of this project
  - Download the two zip files from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
  - Unzip them to create `mnist_test.csv` and `mnist_train.csv` into the directory `data/mnist`
-  Execute the MnistMain.kt file, which will automatically load the dataset, initialize the neural network, train it with the dataset, and evaluate its performance.

### Understanding MnistMain

- Data Preparation: The class includes methods for loading and preparing the MNIST dataset.
- Network Configuration: It demonstrates how to configure the neural network for the task of digit recognition.
- Training and Evaluation: It includes the process of training the network with the MNIST dataset and evaluating its accuracy.

## Overview

### NeuralNetwork

The `NeuralNetwork` class  provides a framework for building and training neural networks in Kotlin. 
It encapsulates core functionalities such as forward propagation through layers with `forward(input: Matrix)`, 
backward propagation with `backward(predicted: Matrix, actual: Matrix, currentLearningRate: Double)`, 
and a training loop `train(inputs: List<Matrix>, targets: List<Matrix>, epochs: Int, batchSize: Int)`. 

It leverages predefined interfaces for loss functions, optimizers, and learning rate adjustments. 

This design allows for flexible neural network configurations and efficient learning through batch processing and learning rate scheduling.


### Layer

The `Layer` interface and `DenseLayer` class define the structure and behavior of neural network layers in Kotlin. 

`Layer` declares `forward(input: Matrix)` and `backward(errorSignal: Matrix, optimizer: Optimizer, learningRate: Double)` methods for forward and backward propagation. 

#### DenseLayer

The `DenseLayer` class represents a fully connected layer, where each neuron in the layer is connected to all neurons in the previous layer. 
This structure is fundamental for many types of neural networks, including deep learning models used for classification, regression, and more complex tasks. 


### Activation

Activation functions take an input signal and produce an output signal, but they take into account the threshold. They are applied to the input of a neuron (or a layer of neurons) and determine whether it should be activated or not, influencing the ability of the network to learn and make predictions.

Each activation function class implements the Activation interface, which requires two methods:

`activation(m: Matrix): Matrix`: Applies the activation function to each element of the input matrix m and returns a new matrix with the results.
`derivativeActivation(input: Matrix, output: Matrix): Matrix`: Computes the derivative of the activation function with respect to the input matrix, which is essential for backpropagation during training.

#### ReLU (Rectified Linear Unit)

Description: Implements the ReLU activation function, which outputs the input directly if it is positive, else it will output zero. It is defined as f(x) = max(0, x).

Usage: Commonly used in hidden layers due to its efficiency and simplicity.

#### Sigmoid

Description: Implements the Sigmoid activation function, defined as f(x) = 1 / (1 + exp(-x)). It outputs a value between 0 and 1, making it suitable for probabilities.

Usage: Often used in the output layer for binary classification problems.

#### Tanh (Hyperbolic Tangent)

Description: Implements the Tanh activation function, which outputs values between -1 and 1. It is defined as f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).

Usage: Useful in hidden layers when the data is centered around zero.

#### Softmax

Description: Implements the Softmax activation function, which is often used in the output layer for multi-class classification problems. It converts logits (raw predictions) into probabilities by taking the exponential of each output and then normalizing these values by dividing by the sum of all the exponentials.

Usage: Typically used in the output layer to represent a probability distribution over multiple classes.


### Optimizer

The `Optimizer` interface defines a method for updating the weights and biases of a neural network layer.

`fun update(weights: Matrix, bias: Matrix, weightGradient: Matrix, biasGradient: Matrix, learningRate: Double): Pair<Matrix, Matrix>`

Parameters:
* `weights`: Current weights of a layer.
* `bias`: Current biases of a layer.
* `weightGradient`: Gradient of the loss function w.r.t. the weights.
* `biasGradient`: Gradient of the loss function w.r.t. the biases.
* `learningRate`: Step size for the weight update.
 
Returns: A pair containing the updated weights and biases.

#### GradientDescentOptimizer

#### MomentumOptimizer

#### AdagradOptimizer

#### RMSpropOptimizer

#### AdamOptimizer


### LearningRate

#### FixedLearningRate

#### StepDecayLearningRate

#### ExponentialDecayLearningRate

### Matrix

The Matrix class provides a comprehensive set of functionalities for creating and manipulating matrices, essential for various numerical computations and operations in neural networks. This documentation covers the core aspects of the Matrix class, including its constructors, methods, and utility functions.

#### Primary Constructor

```kotlin
Matrix(rows: Int, cols: Int, init: (index: Int) -> Double)
```

Parameters:
- `rows`: Number of rows in the matrix.
- `cols`: Number of columns in the matrix.
- `init`: A lambda function to initialize the matrix elements based on their index.

Description: Creates a matrix of the specified size, with each element initialized according to the provided lambda function.

#### Secondary Constructors

```kotlin
Matrix(rows: Int, cols: Int, value: Double)
```

Description: Initializes a matrix where all elements are set to the specified value.

```kotlin
Matrix(rows: Int, cols: Int, values: List<Double>)
```

Description: Initializes a matrix with elements from the provided list. The list size must match rows * cols.

```kotlin
Matrix(rows: Int, cols: Int, init: (row: Int, col: Int) -> Double)
```

Description: Similar to the primary constructor but allows initialization based on row and column indices rather than a flat index.

#### Arithmetic Operations
- Addition (`+`): Adds two matrices or a matrix and a scalar.
- Subtraction (`-`): Subtracts two matrices or a matrix and a scalar.
- Multiplication (`*`): Multiplies two matrices or a matrix by a scalar.
- Division (`/`): Divides matrix elements by another matrix or a scalar.

#### Utility Functions

- `transpose()`: Returns a new matrix that is the transpose of the original matrix.
- `dot(other: Matrix)`: Performs the dot product between two matrices.
- `map(func: (value: Double) -> Double)`: Returns a new matrix with a function applied to each element.
- `sum()`: Calculates the sum of all elements in the matrix.
- `norm()`: Computes the Frobenius norm of the matrix.

#### Element-wise Operations

- `add(other: Matrix)`: Element-wise addition of two matrices.
- `subtract(other: Matrix)`: Element-wise subtraction of two matrices.
- `multiply(other: Matrix)`: Element-wise multiplication of two matrices.
- `divide(other: Matrix)`: Element-wise division of two matrices.

#### Example Usage

```kotlin
val matrix1 = Matrix(2, 2, 1.0)
val matrix2 = Matrix(2, 2) { row, col -> (row + col).toDouble() }
val result = matrix1 + matrix2
```

This example demonstrates creating two matrices and adding them together.
The Matrix class provides a flexible API for performing a wide range of matrix operations, making it a foundational component for numerical computations and neural network implementations.
