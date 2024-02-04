package ch.obermuhlner.neuralnetwork

import java.lang.Math.pow
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

interface LearningRate {
    fun learningRate(epoch: Int): Double
}

class FixedLearningRate(private val learningRate: Double) : LearningRate {
    override fun learningRate(epoch: Int): Double {
        return learningRate
    }
}

class StepDecayLearningRate(
    private val initialLearningRate: Double,
    private val decayFactor: Double,
    private val decayEpochs: Int
) : LearningRate {
    override fun learningRate(epoch: Int): Double {
        return initialLearningRate * decayFactor.pow((epoch / decayEpochs).toDouble())
    }
}

class ExponentialDecayLearningRate(
    private val initialLearningRate: Double,
    private val decayRate: Double
) : LearningRate {
    override fun learningRate(epoch: Int): Double {
        return initialLearningRate * exp(-decayRate * epoch)
    }
}

class AdaGrad(
    private val initialLearningRate: Double,
    private val epsilon: Double = 1e-8
) : LearningRate {
    private var accumulatedGradSquared: Double = 0.0

    override fun learningRate(epoch: Int): Double {
        return initialLearningRate / (sqrt(accumulatedGradSquared) + epsilon)
    }

    fun update(gradient: DoubleArray) {
        accumulatedGradSquared += gradient.sumOf { it * it }
    }
}

class RMSprop(
    private val initialLearningRate: Double,
    private val decayRate: Double = 0.9,
    private val epsilon: Double = 1e-8
) : LearningRate {
    private var movingAvgGradSquared: Double = 0.0

    override fun learningRate(epoch: Int): Double {
        return initialLearningRate / (Math.sqrt(movingAvgGradSquared) + epsilon)
    }

    fun update(gradient: DoubleArray) {
        movingAvgGradSquared = decayRate * movingAvgGradSquared + (1 - decayRate) * gradient.sumOf { it * it }
    }
}

class Adam(
    private val initialLearningRate: Double,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val epsilon: Double = 1e-8
) : LearningRate {
    private var t: Int = 0
    private var m: Double = 0.0
    private var v: Double = 0.0

    override fun learningRate(epoch: Int): Double {
        t++
        val alpha = initialLearningRate * sqrt(1 - beta2.pow(t.toDouble())) / (1 - beta1.pow(t.toDouble()))
        return alpha / (sqrt(v) + epsilon)
    }

    fun update(gradient: DoubleArray) {
        t++
        m = beta1 * m + (1 - beta1) * gradient.sumOf { it }
        v = beta2 * v + (1 - beta2) * gradient.sumOf { it * it }
    }
}
