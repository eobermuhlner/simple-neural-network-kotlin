package ch.obermuhlner.neuralnetwork

import kotlin.math.sqrt
import kotlin.random.Random

interface Layer {
    fun forward(input: Matrix): Matrix
    fun backward(errorSignal: Matrix, optimizer: Optimizer, learningRate: Double): Matrix
}

class DenseLayer(
    private val inputSize: Int,
    private val outputSize: Int,
    private val activationFunc: Activation = ReLU(),
    private val random: Random = Random.Default,
    weightInit: (index: Int) -> Double = { _ -> random.nextDouble() / sqrt(inputSize.toDouble())},
    biasInit: (index: Int) -> Double = { _ -> 0.0 }
) : Layer {
    var weights: Matrix = Matrix(outputSize, inputSize, weightInit)
    var biases: Matrix = Matrix(outputSize, 1, biasInit)

    // For storing the inputs to the layer, and the outputs from the activation function
    private lateinit var lastInput: Matrix
    private lateinit var lastOutput: Matrix

    private lateinit var weightGradients: Matrix
    private lateinit var biasGradients: Matrix

    override fun forward(input: Matrix): Matrix {
        lastInput = input
        val z = weights.dot(input).add(biases) // Compute weighted input plus bias
        lastOutput = activationFunc.activation(z) // Apply activation function
        return lastOutput
    }

    override fun backward(errorSignal: Matrix, optimizer: Optimizer, learningRate: Double): Matrix {
        // Calculate derivative of activation function using both the input to the activation function (lastZ)
        // and the output from the activation function (lastOutput)
        val z = weights.dot(lastInput).add(biases) // Recalculate z to get the input to the activation function
        val sigmaPrime = activationFunc.derivativeActivation(z, lastOutput)

        val delta = errorSignal.multiply(sigmaPrime) // Calculate delta

        // Compute gradients for weights and biases
        weightGradients = delta.dot(lastInput.transpose())
        biasGradients = delta.sumRows() // Sum errors over all examples for each bias

        // Update weights and biases
        val updated = optimizer.update(weights, biases, weightGradients, biasGradients, learningRate)
        weights = updated.first
        biases = updated.second

        // Compute the error signal for the previous layer
        return weights.transpose().dot(delta)
    }
}
