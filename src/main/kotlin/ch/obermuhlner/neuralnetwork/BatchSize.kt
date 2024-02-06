package ch.obermuhlner.neuralnetwork

import kotlin.math.min
import kotlin.math.pow

interface BatchSize {
    fun batchSize(epoch: Int): Int
}

class FixedBatchSize(private val batchSize: Int): BatchSize {
    override fun batchSize(epoch: Int): Int {
        return batchSize
    }
}

class LinearIncreaseBatchSize(private val startSize: Int, private val increment: Int, private val maxSize: Int) : BatchSize {
    override fun batchSize(epoch: Int): Int {
        val size = startSize + increment * (epoch - 1)
        return min(size, maxSize)
    }
}

class ExponentialGrowthBatchSize(private val startSize: Int, private val growthRate: Double, private val maxSize: Int) : BatchSize {
    override fun batchSize(epoch: Int): Int {
        val size = (startSize * growthRate.pow((epoch - 1).toDouble())).toInt()
        return min(size, maxSize)
    }
}

class StepIncreaseBatchSize(private val startSize: Int, private val increment: Int, private val stepSize: Int, private val maxSize: Int) : BatchSize {
    override fun batchSize(epoch: Int): Int {
        val size = startSize + (epoch - 1) / stepSize * increment
        return min(size, maxSize)
    }
}

