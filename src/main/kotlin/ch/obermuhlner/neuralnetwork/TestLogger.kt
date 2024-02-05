package ch.obermuhlner.neuralnetwork

import kotlin.math.max

interface TestLogger {
    fun test(neuralNetwork: NeuralNetwork)
}

class NopTestLogger: TestLogger {
    override fun test(neuralNetwork: NeuralNetwork) {
        // do nothing
    }
}

class ClassificationTestLogger(private val testInputs: List<Matrix>, private val testOutputs: List<Matrix>): TestLogger {
    override fun test(neuralNetwork: NeuralNetwork) {
        val n = max(testOutputs[0].rows, testOutputs[0].cols)
        val confusionMatrix = MutableMatrix(n, n)
        var countCorrect = 0
        for (index in testInputs.indices) {
            val input = testInputs[index]
            val testOutput = testOutputs[index]
            val testOutputLabel = Classification.toOutputLabel(testOutput)

            val output = neuralNetwork.forward(input)
            val outputLabel = Classification.toOutputLabel(output)

            val correct = outputLabel == testOutputLabel
            if (correct) {
                countCorrect += 1
            }

            confusionMatrix[testOutputLabel, outputLabel] += 1.0
            //println("expected = $label, actual = $outputLabel, correct = $correct")
        }

        println()
        val accuracy = countCorrect.toDouble() / testInputs.size

        println("Accuracy: $accuracy")
        println("Confusion:")
        println(confusionMatrix.contentToString(true))
    }
}
