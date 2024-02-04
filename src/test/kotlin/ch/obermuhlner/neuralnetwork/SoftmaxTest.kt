package ch.obermuhlner.neuralnetwork

import org.junit.jupiter.api.Test

class SoftmaxTest {

    @Test
    fun testSoftmax() {
        val softmax = Softmax()

        val result = softmax.activation(Matrix(1, 7, listOf(1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0)))
        println(result.contentToString())
    }
}