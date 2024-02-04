package ch.obermuhlner.neuralnetwork

interface Optimizer {
    fun update(weights: Matrix, bias: Matrix, weightGradient: Matrix, biasGradient: Matrix, learningRate: Double): Pair<Matrix, Matrix>
}

class StochasticGradientDescent : Optimizer {
    override fun update(weights: Matrix, bias: Matrix, weightGradient: Matrix, biasGradient: Matrix, learningRate: Double): Pair<Matrix, Matrix> {
        val updatedWeights = weights.subtract(weightGradient.multiply(learningRate))
        val updatedBias = bias.subtract(biasGradient.multiply(learningRate))
        return Pair(updatedWeights, updatedBias)
    }
}