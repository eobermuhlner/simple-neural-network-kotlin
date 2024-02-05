package ch.obermuhlner.neuralnetwork

import kotlin.math.pow

interface Optimizer {
    fun update(weights: Matrix, bias: Matrix, weightGradient: Matrix, biasGradient: Matrix, learningRate: Double): Pair<Matrix, Matrix>
}

class GradientDescentOptimizer : Optimizer {
    override fun update(
        weights: Matrix,
        bias: Matrix,
        weightGradient: Matrix,
        biasGradient: Matrix,
        learningRate: Double
    ): Pair<Matrix, Matrix> {
        val updatedWeights = weights - (weightGradient * learningRate)
        val updatedBias = bias - (biasGradient * learningRate)

        return Pair(updatedWeights, updatedBias)
    }
}

class MomentumOptimizer(val gamma: Double = 0.9) : Optimizer {
    private var velocityW: Matrix? = null
    private var velocityB: Matrix? = null

    override fun update(weights: Matrix, bias: Matrix, weightGradient: Matrix, biasGradient: Matrix, learningRate: Double): Pair<Matrix, Matrix> {
        if (velocityW == null) {
            velocityW = Matrix(weights.rows, weights.cols)
        }
        if (velocityB == null) {
            velocityB = Matrix(bias.rows, bias.cols)
        }

        velocityW = (velocityW!! * gamma) + (weightGradient * learningRate)
        val updatedWeights = weights - velocityW!!

        velocityB = (velocityB!! * gamma) + (biasGradient * learningRate)
        val updatedBias = bias - velocityB!!

        return Pair(updatedWeights, updatedBias)
    }
}

class AdagradOptimizer(private val epsilon: Double = 1e-8) : Optimizer {
    private var sumSquaredGradientW: Matrix? = null
    private var sumSquaredGradientB: Matrix? = null

    override fun update(weights: Matrix, bias: Matrix, weightGradient: Matrix, biasGradient: Matrix, learningRate: Double): Pair<Matrix, Matrix> {
        if (sumSquaredGradientW == null) {
            sumSquaredGradientW = Matrix(weights.rows, weights.cols)
        }
        if (sumSquaredGradientB == null) {
            sumSquaredGradientB = Matrix(bias.rows, bias.cols)
        }

        sumSquaredGradientW = sumSquaredGradientW!! + weightGradient.pow(2.0)
        val adjustedLearningRateW = (sumSquaredGradientW!!.sqrt() + epsilon).reciprocal() * learningRate
        val updatedWeights = weights - (weightGradient * adjustedLearningRateW)

        sumSquaredGradientB = sumSquaredGradientB!! + biasGradient.pow(2.0)
        val adjustedLearningRateB = (sumSquaredGradientB!!.sqrt() + epsilon).reciprocal() * learningRate
        val updatedBias = bias - (biasGradient * adjustedLearningRateB)

        return Pair(updatedWeights, updatedBias)
    }
}

class RMSpropOptimizer(val decayRate: Double = 0.9, val epsilon: Double = 1e-8) : Optimizer {
    private var squaredGradientW: Matrix? = null
    private var squaredGradientB: Matrix? = null

    override fun update(weights: Matrix, bias: Matrix, weightGradient: Matrix, biasGradient: Matrix, learningRate: Double): Pair<Matrix, Matrix> {
        if (squaredGradientW == null) {
            squaredGradientW = Matrix(weights.rows, weights.cols)
        }
        if (squaredGradientB == null) {
            squaredGradientB = Matrix(bias.rows, bias.cols)
        }

        squaredGradientW = (squaredGradientW!! * decayRate) + (weightGradient.pow(2.0) * (1 - decayRate))
        val updatedWeights = weights - (weightGradient * ((squaredGradientW!!.sqrt() + epsilon).reciprocal() * learningRate))

        squaredGradientB = (squaredGradientB!! * decayRate) + (biasGradient.pow(2.0) * (1 - decayRate))
        val updatedBias = bias - (biasGradient * ((squaredGradientB!!.sqrt() + epsilon).reciprocal() * learningRate))

        return Pair(updatedWeights, updatedBias)
    }
}

class AdamOptimizer(
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val epsilon: Double = 1e-8
) : Optimizer {
    private var mW: Matrix? = null
    private var vW: Matrix? = null
    private var mB: Matrix? = null
    private var vB: Matrix? = null
    private var timestep: Int = 0

    override fun update(weights: Matrix, bias: Matrix, weightGradient: Matrix, biasGradient: Matrix, learningRate: Double): Pair<Matrix, Matrix> {
        // Initialize the first moment vector and the second moment vector if they're null
        if (mW == null) mW = Matrix(weights.rows, weights.cols)
        if (vW == null) vW = Matrix(weights.rows, weights.cols)
        if (mB == null) mB = Matrix(bias.rows, bias.cols)
        if (vB == null) vB = Matrix(bias.rows, bias.cols)

        timestep++

        // Update biased first moment estimate for weights and biases
        mW = mW!! * beta1 + weightGradient * (1 - beta1)
        mB = mB!! * beta1 + biasGradient * (1 - beta1)

        // Update biased second raw moment estimate for weights and biases
        vW = vW!! * beta2 + weightGradient.pow(2.0) * (1 - beta2)
        vB = vB!! * beta2 + biasGradient.pow(2.0) * (1 - beta2)

        // Compute bias-corrected first moment estimate for weights and biases
        val mWCorrected = mW!! * (1 / (1 - beta1.pow(timestep.toDouble())))
        val mBCorrected = mB!! * (1 / (1 - beta1.pow(timestep.toDouble())))

        // Compute bias-corrected second raw moment estimate for weights and biases
        val vWCorrected = vW!! * (1 / (1 - beta2.pow(timestep.toDouble())))
        val vBCorrected = vB!! * (1 / (1 - beta2.pow(timestep.toDouble())))

        // Update weights and biases
        val updatedWeights = weights - (mWCorrected / (vWCorrected.sqrt() + epsilon) * learningRate)
        val updatedBias = bias - (mBCorrected / (vBCorrected.sqrt() + epsilon) * learningRate)

        return Pair(updatedWeights, updatedBias)
    }
}
