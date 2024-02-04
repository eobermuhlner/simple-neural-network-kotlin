package ch.obermuhlner.neuralnetwork

import java.lang.Integer.min

class NeuralNetwork(
    private val lossFunction: LossFunction,
    private val optimizer: Optimizer,
    private val learningRate: LearningRate,
    private val layers: List<Layer>
    ) {

    fun forward(input: Matrix): Matrix {
        var output = input
        layers.forEach { layer ->
            output = layer.forward(output)
        }
        return output
    }

    fun backward(predicted: Matrix, actual: Matrix, currentLearningRate: Double) {
        var errorSignal = lossFunction.gradient(predicted, actual)
        for (layer in layers.asReversed()) {
            errorSignal = layer.backward(errorSignal, optimizer, currentLearningRate)
        }
    }

    fun train(inputs: List<Matrix>, targets: List<Matrix>, epochs: Int, batchSize: Int) {
        for (epoch in 1 .. epochs) {
            val currentLearningRate = learningRate.learningRate(epoch)

            val shuffledIndices = inputs.indices.shuffled()

            for (startIndex in inputs.indices step batchSize) {
                val endIndex = min(startIndex + batchSize, inputs.size)

                for (index in shuffledIndices.slice(startIndex until endIndex)) {
                    train(inputs[index], targets[index], currentLearningRate)
                }
            }
        }
    }

    private fun train(input: Matrix, target: Matrix, currentLearningRate: Double) {
        val predicted = forward(input)
        val loss = lossFunction.loss(predicted, target)
        println("Loss: $loss")
        backward(predicted, target, currentLearningRate)
    }
}
