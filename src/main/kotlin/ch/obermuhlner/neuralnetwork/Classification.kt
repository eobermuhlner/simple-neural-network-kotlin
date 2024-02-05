package ch.obermuhlner.neuralnetwork

object Classification {

    fun toOutputMatrix(label: Int, labelCount: Int): Matrix {
        return Matrix(1, labelCount, {index -> if (index == label) 1.0 else 0.0 })
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
}