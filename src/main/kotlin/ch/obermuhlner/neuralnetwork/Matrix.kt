package ch.obermuhlner.neuralnetwork

import java.lang.StringBuilder

open class Matrix(val rows: Int, val cols: Int, init: (index: Int) -> Double,
             protected val data: DoubleArray = DoubleArray(rows * cols, init)
) {
    constructor(rows: Int, cols: Int, value: Double = 0.0) : this(rows, cols, { _ -> value })

    constructor(rows: Int, cols: Int, values: List<Double>) : this(rows, cols, { index -> values[index] })

    constructor(rows: Int, cols: Int, init: (row: Int, col: Int) -> Double) : this(rows, cols, { index -> init(index / cols, index % cols) })

    val size: Int get() = data.size

    operator fun get(row: Int, col: Int): Double = data[row * cols + col]
    operator fun set(row: Int, col: Int, value: Double) {
        data[row * cols + col] = value
    }

    operator fun get(index: Int): Double = data[index]
    operator fun set(index: Int, value: Double) {
        data[index] = value
    }

    operator fun plus(other: Matrix): Matrix {
        return this.add(other)
    }

    operator fun minus(other: Matrix): Matrix {
        return this.subtract(other)
    }

    operator fun times(other: Matrix): Matrix {
        return this.multiply(other)
    }

    operator fun times(scalar: Double): Matrix {
        return this.multiply(scalar)
    }

    operator fun div(scalar: Double): Matrix {
        return this.divide(scalar)
    }

    fun add(other: Matrix): Matrix {
        if (this.rows != other.rows) throw IllegalArgumentException("Matrix this.rows = ${this.rows} != other.rows = ${other.rows}")
        if (this.cols != other.cols) throw IllegalArgumentException("Matrix this.cols = ${this.cols} != other.cols = ${other.cols}")
        return Matrix(rows, cols, { i -> this.data[i] + other.data[i] })
    }

    fun subtract(other: Matrix): Matrix {
        if (this.rows != other.rows) throw IllegalArgumentException("Matrix this.rows = ${this.rows} != other.rows = ${other.rows}")
        if (this.cols != other.cols) throw IllegalArgumentException("Matrix this.cols = ${this.cols} != other.cols = ${other.cols}")
        return Matrix(rows, cols, { i -> this.data[i] - other.data[i] })
    }

    fun multiply(other: Matrix): Matrix {
        if (this.rows != other.rows) throw IllegalArgumentException("Matrix this.rows = ${this.rows} != other.rows = ${other.rows}")
        if (this.cols != other.cols) throw IllegalArgumentException("Matrix this.cols = ${this.cols} != other.cols = ${other.cols}")
        return Matrix(rows, cols, { i -> this.data[i] * other.data[i] })
    }

    fun multiply(scalar: Double): Matrix {
        return Matrix(rows, cols, { i -> this.data[i] * scalar })
    }

    fun divide(scalar: Double): Matrix {
        return Matrix(rows, cols, { i -> this.data[i] / scalar })
    }

    fun sum(): Double {
        return data.sum()
    }

    fun sumColumns(): Matrix {
        return Matrix(1, cols, { col ->
            var sum = 0.0
            for (row in 0 until this.rows) {
                sum += this[row, col]
            }
            sum
        })
    }

    fun sumRows(): Matrix {
        return Matrix(rows, 1, { row ->
            var sum = 0.0
            for (col in 0 until this.cols) {
                sum += this[row, col]
            }
            sum
        })
    }

    infix fun dot(other: Matrix): Matrix {
        if (this.cols != other.rows) throw IllegalArgumentException("Matrix this.cols = ${this.cols} != other.rows = ${other.rows}")
        val result = Matrix(this.rows, other.cols)
        for (row in 0 until this.rows) {
            for (col in 0 until other.cols) {
                var sum = 0.0
                for (k in 0 until this.cols) {
                    sum += this[row, k] * other[k, col]
                }
                result[row, col] = sum
            }
        }
        return result
    }

    fun map(func: (value: Double) -> Double): Matrix {
        return Matrix(rows, cols, { index -> func(this[index]) })
    }

    fun map(func: (index: Int, value: Double) -> Double): Matrix {
        return Matrix(rows, cols, { index -> func(index, this[index]) })
    }

    fun map(func: (row: Int, col: Int, value: Double) -> Double): Matrix {
        return Matrix(rows, cols) { row, col -> func(row, col, this[row, col]) }
    }

    fun transpose(): Matrix {
        val result = Matrix(this.cols, this.rows)
        for (row in 0 until this.rows) {
            for (col in 0 until this.cols) {
                result[col, row] = this[row, col]
            }
        }
        return result
    }

    override fun toString(): String {
        return "Matrix $rows x $cols"
    }

    fun contentToString(): String {
        val result = StringBuilder()
        result.append("[")
        for (row in 0 until rows) {
            if (row != 0) {
                result.append(",\n")
                result.append(" [")
            } else {
                result.append("[")
            }
            for (col in 0 until cols) {
                if (col != 0) {
                    result.append(", ")
                }
                result.append(this[row, col])
            }
            result.append("]")
        }
        result.append("]")
        return result.toString()
    }
}



class MutableMatrix(rows: Int, cols: Int, init: (Int) -> Double) : Matrix(rows, cols, init) {

    constructor(rows: Int, cols: Int, value: Double = 0.0) : this(rows, cols, { _ -> value })

    constructor(rows: Int, cols: Int, init: (row: Int, col: Int) -> Double) : this(rows, cols, { index -> init(index / cols, index % cols) })

    fun addInplace(other: Matrix) {
        if (this.rows != other.rows) throw IllegalArgumentException("Matrix this.rows = ${this.rows} != other.rows = ${other.rows}")
        if (this.cols != other.cols) throw IllegalArgumentException("Matrix this.cols = ${this.cols} != other.cols = ${other.cols}")
        for (i in data.indices) {
            this.data[i] += other[i]
        }
    }

    fun subtractInplace(other: Matrix) {
        if (this.rows != other.rows) throw IllegalArgumentException("Matrix this.rows = ${this.rows} != other.rows = ${other.rows}")
        if (this.cols != other.cols) throw IllegalArgumentException("Matrix this.cols = ${this.cols} != other.cols = ${other.cols}")
        for (i in data.indices) {
            this.data[i] -= other[i]
        }
    }

    fun multiplyInplace(scalar: Double) {
        for (i in data.indices) {
            this.data[i] *= scalar
        }
    }

    fun forEach(func: (row: Int, col: Int, value: Double) -> Double) {
        for (row in 0 until this.rows) {
            for (col in 0 until this.cols) {
                this[row, col] = func(row, col, this[row, col])
            }
        }
    }

    fun forEach(func: (value: Double) -> Double): Matrix {
        for (i in data.indices) {
            this.data[i] = func(data[i])
        }
        return this
    }
}
