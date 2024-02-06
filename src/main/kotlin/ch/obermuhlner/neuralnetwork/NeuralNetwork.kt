package ch.obermuhlner.neuralnetwork

import java.lang.Integer.min

class NeuralNetwork(
    private val lossFunction: LossFunction,
    private val optimizer: Optimizer,
    private val gradientClip: GradientClip,
    private val learningRate: LearningRate,
    private val batchSize: BatchSize,
    private val layers: List<Layer>,
    private val epochTestLogger: TestLogger = NopTestLogger(),
    private val batchTestLogger: TestLogger = NopTestLogger()
    ) {

    fun forward(input: Matrix): Matrix {
        var output = input
        layers.forEach { layer ->
            output = layer.forward(output)
        }
        return output
    }

    fun backward(predicted: Matrix, actual: Matrix, currentLearningRate: Double) {
        var errorSignal = gradientClip.clip(lossFunction.gradient(predicted, actual))
        for (layer in layers.asReversed()) {
            errorSignal = layer.backward(errorSignal, optimizer, currentLearningRate)
        }
    }

    fun train(inputs: List<Matrix>, targets: List<Matrix>, epochs: Int) {
        for (epoch in 1 .. epochs) {
            val currentLearningRate = learningRate.learningRate(epoch)
            val currentBatchSize = batchSize.batchSize(epoch)
            val shuffledIndices = inputs.indices.shuffled()

            println("Epoch: $epoch, LearningRate: $currentLearningRate, BatchSize: $currentBatchSize")

            for (startIndex in inputs.indices step currentBatchSize) {
                val endIndex = min(startIndex + currentBatchSize, inputs.size)

                val batchTargets = mutableListOf<Matrix>()
                val batchPredicted = mutableListOf<Matrix>()

                for (index in shuffledIndices.slice(startIndex until endIndex)) {
                    val input = inputs[index]
                    val target = targets[index]
                    val predicted = forward(input)

                    batchTargets.add(target)
                    batchPredicted.add(predicted)
                }

                for ((predicted, target) in batchPredicted.zip(batchTargets)) {
                    backward(predicted, target, currentLearningRate)
                }
                batchTestLogger.test(this)
            }
            epochTestLogger.test(this)
        }
    }

    private fun train(input: Matrix, target: Matrix, currentLearningRate: Double) {
        val predicted = forward(input)
        //val loss = lossFunction.loss(predicted, target)
        //println("Loss: $loss")
        backward(predicted, target, currentLearningRate)
    }
}
