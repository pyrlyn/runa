package runa.gguf

/** The extension of every model file runa handles. */
public const val GGUF_EXTENSION: String = ".gguf"

/** `name-00001-of-00003.gguf`, the naming of llama.cpp's `gguf-split`. */
public data class GgufSplitPart(
    /** The file name without the part suffix and the extension. */
    val stem: String,
    val index: Int,
    val total: Int,
) {
    /** True when [index] is one of the [total] parts; `-00000-of-00000` or `-00003-of-00002` is not. */
    public val isValid: Boolean get() = total >= 1 && index in 1..total

    /** The file name of part [index] of the same model. */
    public fun fileName(index: Int): String = "$stem-${pad(index)}-of-${pad(total)}$GGUF_EXTENSION"

    private fun pad(number: Int) = number.toString().padStart(5, '0')

    public companion object {
        private val PATTERN = Regex("(.+)-(\\d{5})-of-(\\d{5})\\.gguf", RegexOption.IGNORE_CASE)

        /** Null when [fileName] is not a part of a split model. */
        public fun parse(fileName: String): GgufSplitPart? =
            PATTERN.matchEntire(fileName)?.destructured?.let { (stem, index, total) -> GgufSplitPart(stem, index.toInt(), total.toInt()) }

        /**
         * True when [indices], the part numbers of files that share a stem and a total, are
         * exactly 1 to [total]: a set like 1 and 3 of 2 is the right size and still incomplete.
         */
        public fun isComplete(
            indices: Collection<Int>,
            total: Int,
        ): Boolean = total >= 1 && indices.size == total && indices.toSet() == (1..total).toSet()
    }
}

/** True for a visible file with the `.gguf` extension. */
public fun isGgufFileName(fileName: String): Boolean = fileName.endsWith(GGUF_EXTENSION, ignoreCase = true) && !fileName.startsWith(".")

/** The file name without `.gguf` and without a split-part suffix. */
public fun ggufStem(fileName: String): String = GgufSplitPart.parse(fileName)?.stem ?: fileName.dropLast(GGUF_EXTENSION.length)

// The quantization tag that converters put at the end of a file name: `-Q4_K_M`, `.IQ3_XXS`, `-f16`.
private val QUANTIZATION_SUFFIX = Regex("[-._]((?:I?Q\\d|TQ\\d|BF\\d|F\\d)[A-Z0-9_]*)$", RegexOption.IGNORE_CASE)

/** The quantization named at the end of a file stem, in upper case; for when the metadata is not at hand. */
public fun quantizationInName(stem: String): String? =
    QUANTIZATION_SUFFIX
        .find(stem)
        ?.groupValues
        ?.get(1)
        ?.uppercase()

/** [stem] without its quantization tag: the model's own name. */
public fun withoutQuantization(stem: String): String = stem.replace(QUANTIZATION_SUFFIX, "")
