import ch.obermuhlner.neuralnetwork.Matrix
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import kotlin.math.sqrt

class MatrixTest {

    @Test
    fun testMatrixCreation() {
        val matrix = Matrix(2, 3, 5.0)
        assertEquals(2, matrix.rows)
        assertEquals(3, matrix.cols)
        for (row in 0 until matrix.rows) {
            for (col in 0 until matrix.cols) {
                assertEquals(5.0, matrix[row, col], "Matrix initialization with constant failed at [$row, $col]")
            }
        }
    }

    @Test
    fun testMatrixAddition() {
        val matrix1 = Matrix(2, 2, 1.0)
        val matrix2 = Matrix(2, 2, 2.0)
        val result = matrix1 + matrix2
        assertEquals(3.0, result[0, 0])
        assertEquals(3.0, result[0, 1])
        assertEquals(3.0, result[1, 0])
        assertEquals(3.0, result[1, 1])
    }

    @Test
    fun testMatrixSubtraction() {
        val matrix1 = Matrix(2, 2, 5.0)
        val matrix2 = Matrix(2, 2, 3.0)
        val result = matrix1 - matrix2
        assertEquals(2.0, result[0, 0])
        assertEquals(2.0, result[1, 1])
    }

    @Test
    fun testMatrixMultiplication() {
        val matrix1 = Matrix(2, 3, 1.0)
        val matrix2 = Matrix(3, 2, 2.0)
        val result = matrix1 dot matrix2
        assertEquals(6.0, result[0, 0])
        assertEquals(6.0, result[1, 1])
    }

    @Test
    fun testMatrixScalarOperations() {
        val matrix = Matrix(2, 2, 2.0)
        val added = matrix + 3.0
        val subtracted = matrix - 1.0
        val multiplied = matrix * 2.0
        val divided = matrix / 2.0

        assertEquals(5.0, added[0, 0])
        assertEquals(1.0, subtracted[0, 0])
        assertEquals(4.0, multiplied[0, 0])
        assertEquals(1.0, divided[0, 0])
    }

    @Test
    fun testMatrixTranspose() {
        val matrix = Matrix(2, 3) { row, col -> (row * 3 + col).toDouble() }
        val transposed = matrix.transpose()
        assertEquals(matrix.rows, transposed.cols)
        assertEquals(matrix.cols, transposed.rows)
        assertEquals(0.0, transposed[0, 0])
        assertEquals(1.0, transposed[1, 0])
        assertEquals(3.0, transposed[0, 1])
    }

    @Test
    fun testMatrixElementWiseMultiplication() {
        val matrix1 = Matrix(2, 2, 3.0)
        val matrix2 = Matrix(2, 2, 2.0)
        val result = matrix1.multiply(matrix2)
        assertEquals(6.0, result[0, 0])
        assertEquals(6.0, result[1, 1])
    }

    @Test
    fun testMatrixElementWiseDivision() {
        val matrix1 = Matrix(2, 2, 4.0)
        val matrix2 = Matrix(2, 2, 2.0)
        val result = matrix1.divide(matrix2)
        assertEquals(2.0, result[0, 0])
        assertEquals(2.0, result[1, 1])
    }

    @Test
    fun testMatrixPower() {
        val matrix = Matrix(2, 2, 2.0)
        val result = matrix.pow(3.0)
        assertEquals(8.0, result[0, 0])
        assertEquals(8.0, result[1, 1])
    }

    @Test
    fun testMatrixSum() {
        val matrix = Matrix(2, 2) { _, _ -> 1.0 }
        val sum = matrix.sum()
        assertEquals(4.0, sum)
    }

    @Test
    fun testMatrixNorm() {
        val matrix = Matrix(2, 2) { _, _ -> 3.0 }
        val norm = matrix.norm()
        assertEquals(sqrt(36.0), norm, 0.001)
    }

    @Test
    fun testMatrixMapFunction() {
        val matrix = Matrix(2, 2) { row, col -> (row + col).toDouble() }
        val result = matrix.map { v -> v * 2 }
        assertEquals(0.0, result[0, 0])
        assertEquals(2.0, result[0, 1])
        assertEquals(2.0, result[1, 0])
        assertEquals(4.0, result[1, 1])
    }

    @Test
    fun testInvalidDimensions() {
        val matrix1 = Matrix(2, 3, 1.0)
        val matrix2 = Matrix(3, 2, 2.0)
        assertThrows(IllegalArgumentException::class.java) {
            matrix1 + matrix2 // Should throw due to mismatched dimensions
        }
    }

    @Test
    fun testMatrixRepeatRows() {
        val matrix = Matrix(1, 2, listOf(1.0, 2.0))
        val repeated = matrix.repeatRows(3)
        assertEquals(3, repeated.rows)
        assertEquals(2, repeated.cols)
        for (row in 0 until repeated.rows) {
            assertEquals(1.0, repeated[row, 0])
            assertEquals(2.0, repeated[row, 1])
        }
    }
}
