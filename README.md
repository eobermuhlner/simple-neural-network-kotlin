Simple implementation of a neural network in Kotlin.

No dependencies to other libraries.

## Overview

### NeuralNetwork

The `NeuralNetwork` class  provides a framework for building and training neural networks in Java. 
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

`Activation` Interface: Specifies two functions for any activation implementation: 
`activation(m: Matrix)`: Matrix to compute the activation of input matrix m, 
and `derivativeActivation(input: Matrix, output: Matrix): Matrix` to calculate the derivative of the activation function given input and output matrices.

#### ReLU

`ReLU` Class: Implements the Rectified Linear Unit (ReLU) activation function. 
The activation method outputs the input value if it's greater than 0; otherwise, it outputs 0. 
Its derivativeActivation method returns 1 for inputs greater than 0, and 0 otherwise, reflecting the gradient of ReLU.

#### Sigmoid

`Sigmoid` Class: Implements the Sigmoid activation function, which outputs values in the range (0,1), useful for binary classification tasks.
The derivative is calculated based on the output of the sigmoid function itself, reflecting its gradient.

#### Tanh

`Tanh` Class: Implements the hyperbolic tangent activation function that outputs values in the range (-1,1).
The derivative is calculated as 1 - tanh^2(value), which is the gradient of the tanh function.

#### Softmax

`Softmax` Class: Used for multi-class classification tasks, it normalizes the input into a probability distribution across various classes. 
The activation method computes the softmax of the input matrix. 
The derivativeActivation method computes the derivative of the softmax function, useful for backpropagation in neural networks.


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