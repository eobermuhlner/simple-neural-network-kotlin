package ch.obermuhlner.neuralnetwork

import java.io.File

// https://www.kaggle.com/datasets/oddrationale/mnist-in-csv


fun main() {
    val inputSize = 28*28
    val hiddenSize = 28
    val outputSize = 10

    val trainMnist = MnistLoader(File("data/mnist/mnist_train.csv")).load()
    val testMnist = MnistLoader(File("data/mnist/mnist_test.csv")).load()

    val trainOutputs = trainMnist.map { Classification.toOutputMatrix(it.first, outputSize) }
    val trainInputs = trainMnist.map { it.second }

    val testOutputs = testMnist.map { Classification.toOutputMatrix(it.first, outputSize) }
    val testInputs = testMnist.map { it.second }

    val network = NeuralNetwork(
        MeanSquareError(),
        GradientDescent(),
        ExponentialDecayLearningRate(0.001, 0.01),
        listOf(
            DenseLayer(inputSize, hiddenSize, ReLU()),
            DenseLayer(hiddenSize, outputSize, ReLU())
        ),
        epochTestLogger = ClassificationTestLogger(testInputs, testOutputs)
    )

    network.train(trainInputs, trainOutputs, 10, 1000)
}

class MnistLoader(private val file: File) {
    fun load(): List<Pair<Int, Matrix>> {
        val result = mutableListOf<Pair<Int, Matrix>>()
        for (line in file.readLines()) {
            if (line.startsWith("label")) {
                // ignore
            } else {
                val values = line.split(",")
                val label = values[0].toInt()
                val matrix = Matrix(28*28, 1, { index -> values[index+1].toDouble() / 256.0})
                result.add(Pair(label, matrix))
            }
        }
        return result
    }
}