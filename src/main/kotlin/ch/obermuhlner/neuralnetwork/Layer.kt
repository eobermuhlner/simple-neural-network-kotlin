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
    var weights: Matrix = Matrix(inputSize, outputSize, weightInit)
    var biases: Matrix = Matrix(1, outputSize, biasInit)

    private lateinit var lastInput: Matrix
    private lateinit var lastOutput: Matrix

    override fun forward(input: Matrix): Matrix {
        lastInput = input
        val z = input.dot(weights) + biases.repeatRows(input.rows) // Adjust biases for batch size
        lastOutput = activationFunc.activation(z)
        return lastOutput
    }

    override fun backward(errorSignal: Matrix, optimizer: Optimizer, learningRate: Double): Matrix {
        val z = lastInput.dot(weights) + biases.repeatRows(lastInput.rows) // Recalculate z if needed
        val sigmaPrime = activationFunc.derivativeActivation(z, lastOutput)

        val delta = errorSignal.multiply(sigmaPrime)

        // Averaging the gradients across the batch for weights
        val weightGradients = lastInput.transpose().dot(delta) / lastInput.rows.toDouble()
        // Summing the gradients across the batch for biases
        val biasGradients = delta.sumColumns() / lastInput.rows.toDouble()

        val updated = optimizer.update(weights, biases, weightGradients, biasGradients, learningRate)
        weights = updated.first
        biases = updated.second

        return delta.dot(weights.transpose())
    }
}

