package ch.obermuhlner.neuralnetwork

import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.tanh

interface Activation {
    fun activation(m: Matrix): Matrix

    fun derivativeActivation(input: Matrix, output: Matrix): Matrix
}

class ReLU : Activation {
    override fun activation(m: Matrix): Matrix = m.map { v -> if (v > 0) v else 0.0 }

    override fun derivativeActivation(input: Matrix, output: Matrix): Matrix = input.map { v -> if (v > 0) 1.0 else 0.0 }
}

class Sigmoid : Activation {
    override fun activation(m: Matrix): Matrix = m.map { value -> 1.0 / (1.0 + exp(-value)) }

    override fun derivativeActivation(input: Matrix, output: Matrix): Matrix = input.map { value ->
        val sigmoid = 1.0 / (1.0 + exp(-value))
        sigmoid * (1 - sigmoid)
    }
}

class Tanh : Activation {
    override fun activation(m: Matrix): Matrix = m.map { value -> tanh(value) }

    override fun derivativeActivation(input: Matrix, output: Matrix): Matrix = input.map { value -> 1 - tanh(value).pow(2.0) }
}

class Softmax : Activation {
    override fun activation(m: Matrix): Matrix {
        val result = MutableMatrix(m.rows, m.cols)
        for (col in 0 until m.cols) {
            var maxVal = Double.NEGATIVE_INFINITY
            for (row in 0 until m.rows) {
                maxVal = maxOf(maxVal, m[row, col])
            }
            var sum = 0.0
            for (row in 0 until m.rows) {
                val expVal = exp(m[row, col] - maxVal)
                result[row, col] = expVal
                sum += expVal
            }
            for (row in 0 until m.rows) {
                result[row, col] = result[row, col] / sum
            }
        }
        return result
    }

    fun activation_ORIG(m: Matrix): Matrix {
        val result = MutableMatrix(m.rows, m.cols)
        for (row in 0 until m.rows) {
            var maxVal = Double.NEGATIVE_INFINITY
            for (col in 0 until m.cols) {
                maxVal = maxOf(maxVal, m[row, col])
            }
            var sum = 0.0
            for (col in 0 until m.cols) {
                val expVal = exp(m[row, col] - maxVal)
                result[row, col] = expVal
                sum += expVal
            }
            for (col in 0 until m.cols) {
                result[row, col] = result[row, col] / sum
            }
        }
        return result
    }

    override fun derivativeActivation(input: Matrix, output: Matrix): Matrix {
        val derivative = MutableMatrix(input.rows, input.cols)
        for (i in 0 until input.rows) {
            for (j in 0 until input.cols) {
                val softmaxOutput = output[i, j]
                derivative[i, j] = softmaxOutput * (1 - softmaxOutput)
            }
        }
        return derivative
    }
}

class CrossEntropyActivation : Activation {
    override fun activation(m: Matrix): Matrix {
        val expValues = m.map { v -> exp(v) }
        val sumExp = expValues.sum()
        return expValues.divide(sumExp)
    }

    override fun derivativeActivation(input: Matrix, output: Matrix): Matrix {
        val m = input.rows
        val n = input.cols
        val result = MutableMatrix(m, n)

        for (i in 0 until m) {
            for (j in 0 until n) {
                result[i, j] = output[i, j] * (1.0 - output[i, j])
            }
        }

        return result
    }
}
