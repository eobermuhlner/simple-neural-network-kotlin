package ch.obermuhlner.neuralnetwork

interface GradientClip {
    fun clip(gradient: Matrix): Matrix
}

class NopGradientClip: GradientClip {
    override fun clip(gradient: Matrix): Matrix {
        return gradient
    }
}

class RangeGradientClip(private val minValue: Double, private val maxValue: Double) : GradientClip {
    override fun clip(gradient: Matrix): Matrix {
        return gradient.map { v -> v.coerceIn(minValue, maxValue) }
    }
}

class NormGradientClip(private val maxNorm: Double = 5.0) : GradientClip {
    override fun clip(gradient: Matrix): Matrix {
        val norm = gradient.norm()
        return if (norm > maxNorm) {
            gradient * (maxNorm / norm)
        } else gradient
    }
}


