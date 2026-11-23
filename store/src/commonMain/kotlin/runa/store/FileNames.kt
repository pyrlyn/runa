package runa.store

/** `name-00001-of-00003.gguf`, the naming of llama.cpp's `gguf-split`. */
internal data class SplitPart(
    /** The file name without the part suffix and the extension. */
    val stem: String,
    val index: Int,
    val total: Int,
) {
    fun fileName(index: Int): String = "$stem-${pad(index)}-of-${pad(total)}$EXTENSION"

    private fun pad(number: Int) = number.toString().padStart(5, '0')

    companion object {
        private val PATTERN = Regex("(.+)-(\\d{5})-of-(\\d{5})\\.gguf", RegexOption.IGNORE_CASE)

        fun parse(fileName: String): SplitPart? =
            PATTERN.matchEntire(fileName)?.destructured?.let { (stem, index, total) -> SplitPart(stem, index.toInt(), total.toInt()) }
    }
}

internal const val EXTENSION: String = ".gguf"

internal fun isGgufName(fileName: String): Boolean = fileName.endsWith(EXTENSION, ignoreCase = true) && !fileName.startsWith(".")

/** The file name without `.gguf` and without a split-part suffix. */
internal fun modelStem(fileName: String): String = SplitPart.parse(fileName)?.stem ?: fileName.dropLast(EXTENSION.length)

// The quantization tag that converters put at the end of a file name: `-Q4_K_M`, `.IQ3_XXS`, `-f16`.
private val QUANTIZATION_SUFFIX = Regex("[-._]((?:I?Q\\d|TQ\\d|BF\\d|F\\d)[A-Z0-9_]*)$", RegexOption.IGNORE_CASE)

/** The quantization named by a file stem, in upper case, when the file's metadata does not say. */
internal fun quantizationOf(stem: String): String? =
    QUANTIZATION_SUFFIX
        .find(stem)
        ?.groupValues
        ?.get(1)
        ?.uppercase()

internal fun withoutQuantization(stem: String): String = stem.replace(QUANTIZATION_SUFFIX, "")
