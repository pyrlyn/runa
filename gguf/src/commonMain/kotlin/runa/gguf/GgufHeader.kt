package runa.gguf

/** Everything in a GGUF file before the weights: the metadata and the tensor list. */
public class GgufHeader(
    public val version: Int,
    public val metadata: Map<String, GgufValue>,
    public val tensors: List<GgufTensor>,
) {
    public fun string(key: String): String? = (metadata[key] as? GgufValue.Text)?.value

    public fun long(key: String): Long? = (metadata[key] as? GgufValue.Integer)?.value

    /** `general.architecture`, for example `llama` or `qwen2`; it prefixes the model's own keys. */
    public val architecture: String? get() = string("general.architecture")

    public val name: String? get() = string("general.name")

    /** `general.type`: `model` (also the meaning of null), `adapter`, `mmproj` and so on. */
    public val type: String? get() = string("general.type")

    public val organization: String? get() = string("general.organization")

    /** The number of weights, counted from the tensor list. */
    public val parameterCount: Long get() = tensors.sumOf { it.elementCount }

    /** The quantization name from `general.file_type`, for example `Q4_K_M`; null when unset or unknown. */
    public val quantization: String? get() = long("general.file_type")?.let { FILE_TYPES[it and GUESSED_FLAG.inv()] }

    /** The context length the model was trained with. */
    public val contextLength: Long? get() = architectureLong("context_length")

    public val embeddingLength: Long? get() = architectureLong("embedding_length")

    public val chatTemplate: String? get() = string("tokenizer.chat_template")

    /**
     * True for models that produce embeddings rather than text. GGUF has no field for this, so it
     * follows llama.cpp's convention: such models set a pooling type, and the BERT family is
     * encoder-only.
     */
    public val isEmbeddingModel: Boolean
        get() = (architectureLong("pooling_type") ?: 0L) > 0L || architecture.orEmpty().contains("bert")

    private fun architectureLong(key: String): Long? = architecture?.let { long("$it.$key") }

    public companion object {
        /** The [type] of a file that holds a whole model. */
        public const val TYPE_MODEL: String = "model"

        // Set by llama.cpp when it had to guess the file type from the tensors.
        private const val GUESSED_FLAG: Long = 1024

        // `enum llama_ftype` in llama.cpp's include/llama.h (checked at v1.0.0), without the
        // LLAMA_FTYPE_ and MOSTLY_ prefixes. An id missing here only costs the QUANT column.
        private val FILE_TYPES: Map<Long, String> =
            mapOf(
                0L to "F32",
                1L to "F16",
                2L to "Q4_0",
                3L to "Q4_1",
                7L to "Q8_0",
                8L to "Q5_0",
                9L to "Q5_1",
                10L to "Q2_K",
                11L to "Q3_K_S",
                12L to "Q3_K_M",
                13L to "Q3_K_L",
                14L to "Q4_K_S",
                15L to "Q4_K_M",
                16L to "Q5_K_S",
                17L to "Q5_K_M",
                18L to "Q6_K",
                19L to "IQ2_XXS",
                20L to "IQ2_XS",
                21L to "Q2_K_S",
                22L to "IQ3_XS",
                23L to "IQ3_XXS",
                24L to "IQ1_S",
                25L to "IQ4_NL",
                26L to "IQ3_S",
                27L to "IQ3_M",
                28L to "IQ2_S",
                29L to "IQ2_M",
                30L to "IQ4_XS",
                31L to "IQ1_M",
                32L to "BF16",
                36L to "TQ1_0",
                37L to "TQ2_0",
                38L to "MXFP4_MOE",
                39L to "NVFP4",
                40L to "Q1_0",
                41L to "Q2_0",
                42L to "F8_E4M3",
            )
    }
}
