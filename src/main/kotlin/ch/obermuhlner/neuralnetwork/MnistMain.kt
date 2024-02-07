package ch.obermuhlner.neuralnetwork

import java.io.File
import kotlin.random.Random

// Download two zip files from the following web page:
// https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
// - mnist_train.csv.zip
// - mnist_test.csv.zip
// Create directory `data/mnist` in the root of this project
// Extract the two *.csv files from the zip files into this directory
// The resulting structure will be:
// - simple-neural-network-kotlin
//   - data
//     - mnist
//       - mnist_train.csv
//       - mnist_test.csv

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
        GradientDescentOptimizer(),
        NopGradientClip(), //NormGradientClip(5.0),
        FixedLearningRate(0.001), //ExponentialDecayLearningRate(0.001, 0.01),
        FixedBatchSize(1),
        listOf(
            DenseLayer(inputSize, hiddenSize, ReLU()),
            DenseLayer(hiddenSize, outputSize, ReLU())
        ),
        epochTestLogger = ClassificationTestLogger(testInputs, testOutputs)
    )

    network.train(trainInputs, trainOutputs, 10)
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
                val matrix = Matrix(1, 28*28, { index -> values[index+1].toDouble() / 255.0})
                result.add(Pair(label, matrix))
            }
        }
        return result
    }
}