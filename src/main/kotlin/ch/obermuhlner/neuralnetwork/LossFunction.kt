package ch.obermuhlner.neuralnetwork

import kotlin.math.ln

interface LossFunction {
    fun loss(predicted: Matrix, actual: Matrix): Double
    fun gradient(predicted: Matrix, actual: Matrix): Matrix
}

class MeanSquareError : LossFunction {
    override fun loss(predicted: Matrix, actual: Matrix): Double {
        var sum = 0.0
        for (i in 0 until predicted.rows) {
            for (j in 0 until predicted.cols) {
                val diff = predicted[i, j] - actual[i, j]
                sum += diff * diff
            }
        }
        return sum / (predicted.rows * predicted.cols)
    }

    override fun gradient(predicted: Matrix, actual: Matrix): Matrix {
        val gradient = MutableMatrix(predicted.rows, predicted.cols)
        for (i in 0 until predicted.rows) {
            for (j in 0 until predicted.cols) {
                // Derivative of MSE with respect to the predicted value
                gradient[i, j] = 2 * (predicted[i, j] - actual[i, j]) / (predicted.rows * predicted.cols)
            }
        }
        return gradient
    }
}

class CrossEntropyLoss : LossFunction {
    override fun loss(predicted: Matrix, actual: Matrix): Double {
        // Ensure dimensions match
        require(predicted.rows == actual.rows && predicted.cols == actual.cols) {
            "Input matrices must have the same dimensions"
        }

        val m = predicted.rows
        val n = predicted.cols

        // Calculate cross-entropy loss
        var totalLoss = 0.0

        for (i in 0 until m) {
            for (j in 0 until n) {
                totalLoss += -actual[i, j] * ln(predicted[i, j] + 1e-15) // Adding a small epsilon to avoid log(0)
            }
        }

        return totalLoss / m.toDouble() // Average loss over examples
    }

    override fun gradient(predicted: Matrix, actual: Matrix): Matrix {
        // Ensure dimensions match
        require(predicted.rows == actual.rows && predicted.cols == actual.cols) {
            "Input matrices must have the same dimensions"
        }

        val m = predicted.rows
        val n = predicted.cols

        // Calculate gradient of cross-entropy loss
        val gradient = MutableMatrix(m, n)

        for (i in 0 until m) {
            for (j in 0 until n) {
                gradient[i, j] = (predicted[i, j] - actual[i, j]) / m.toDouble()
            }
        }

        return gradient
    }
}
