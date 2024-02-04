package ch.obermuhlner.neuralnetwork

import java.io.File

// https://www.kaggle.com/datasets/oddrationale/mnist-in-csv


fun main() {
    val inputSize = 28*28
    val hiddenSize = 28
    val outputSize = 10

    val network = NeuralNetwork(
        MeanSquareError(),
        StochasticGradientDescent(),
        FixedLearningRate(0.001),
        listOf(
            DenseLayer(inputSize, hiddenSize, ReLU()),
            DenseLayer(hiddenSize, outputSize, ReLU())
        )
    )

    val trainMnist = MnistLoader(File("data/mnist/mnist_train.csv")).load()
    val testMnist = MnistLoader(File("data/mnist/mnist_test.csv")).load()

    val trainOutputs = trainMnist.map { toOutputMatrix(it.first) }
    val trainInputs = trainMnist.map { it.second }

    network.train(trainInputs, trainOutputs, 1, 100)

    var countCorrect = 0
    for ((label, input) in testMnist) {
        val output = network.forward(input)
        val outputLabel = toOutputLabel(output)
        val correct = label == outputLabel
        if (correct) {
            countCorrect += 1
        }
        println("expected = $label, actual = $outputLabel, correct = $correct")
    }
    println("Correct ${countCorrect.toDouble() / testMnist.size}")
}

fun toOutputMatrix(label: Int): Matrix {
    return Matrix(10, 1, {index -> if (index == label) 1.0 else 0.0 })
}

fun toOutputLabel(m: Matrix): Int {
    var result = 0
    var resultValue = 0.0
    for (index in 0 until m.size) {
        val value = m[index]
        if (value > resultValue) {
            result = index
            resultValue = value
        }
    }
    return result
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