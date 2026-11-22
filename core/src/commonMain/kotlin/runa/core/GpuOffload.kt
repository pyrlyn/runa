package runa.core

/** How much of a model goes to the GPU: the `--gpu off|max|0..1` option. */
public sealed interface GpuOffload {
    /** The text form used by flags, `config.json` and the API. */
    public val id: String

    public data object Off : GpuOffload {
        override val id: String = "off"
    }

    public data object Max : GpuOffload {
        override val id: String = "max"
    }

    /** The share of layers on the GPU, from 0 to 1. */
    public data class Ratio(
        val value: Double,
    ) : GpuOffload {
        override val id: String get() = value.toString()
    }

    public companion object {
        public fun parse(text: String): GpuOffload? =
            when (text) {
                Off.id -> Off
                Max.id -> Max
                else -> text.toDoubleOrNull()?.takeIf { it in 0.0..1.0 }?.let(::Ratio)
            }
    }
}
