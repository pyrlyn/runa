package runa.hub

import runa.gguf.GgufSplitPart
import runa.gguf.ggufStem
import runa.gguf.isGgufFileName
import runa.gguf.quantizationInName

/** A file in a Hub repository. */
public data class HubFile(
    /** The path inside the repository; the last segment is the file name. */
    val path: String,
    val size: Long,
    /** The SHA-256 of the content in lower-case hex; null for small files that are not stored in LFS. */
    val sha256: String?,
) {
    val name: String get() = path.substringAfterLast('/')
}

/** One downloadable model of a repository: a GGUF file, or all the parts of a split one. */
public data class HubVariant(
    /** In upper case, taken from the file name; null when the name does not say. */
    val quantization: String?,
    /** In part order for a split model. */
    val files: List<HubFile>,
) {
    val sizeBytes: Long get() = files.sumOf { it.size }

    /** What the user types after `@` and sees in lists. */
    val label: String get() = quantization ?: ggufStem(files.first().name)
}

/**
 * Groups the GGUF files of a repository into variants, smallest first. Vision projectors
 * (`mmproj-*.gguf`) are not models and are left out, as are split models with missing parts.
 */
public fun variantsOf(files: List<HubFile>): List<HubVariant> =
    files
        .filter { isGgufFileName(it.name) && !it.name.contains("mmproj", ignoreCase = true) }
        .groupBy { file ->
            GgufSplitPart.parse(file.name)?.let { "${file.path.substringBeforeLast('/', "")}/${it.stem}/${it.total}" }
                ?: file.path
        }.values
        .mapNotNull { parts ->
            val sorted = parts.sortedBy { it.name }
            val split = sorted.mapNotNull { GgufSplitPart.parse(it.name) }
            val complete = split.isEmpty() || GgufSplitPart.isComplete(split.map { it.index }, split.first().total)
            if (complete) HubVariant(quantizationInName(ggufStem(sorted.first().name)), sorted) else null
        }.sortedBy { it.sizeBytes }

/**
 * The default variant for a machine with [memoryBytes] of memory: the largest quantized one whose
 * files take at most [MEMORY_SHARE_PERCENT] percent of it, which leaves room for the context and
 * for everything else that runs. Full-precision files (F16, F32, BF16) are picked only when the
 * repository has nothing else: they cost two to four times the memory for no visible gain. When
 * nothing fits, or the memory is unknown, the smallest variant is returned.
 */
public fun defaultVariant(
    variants: List<HubVariant>,
    memoryBytes: Long?,
): HubVariant? {
    val quantized = variants.filter { it.quantization !in FULL_PRECISION }.ifEmpty { variants }
    val smallest = quantized.minByOrNull { it.sizeBytes } ?: return null
    return quantized.filter { fits(it, memoryBytes) }.maxByOrNull { it.sizeBytes } ?: smallest
}

/** Whether [variant] leaves enough of [memoryBytes] free to be worth loading; false when unknown. */
public fun fits(
    variant: HubVariant,
    memoryBytes: Long?,
): Boolean = memoryBytes != null && variant.sizeBytes <= memoryBytes / 100 * MEMORY_SHARE_PERCENT

public const val MEMORY_SHARE_PERCENT: Int = 70

private val FULL_PRECISION = setOf("F16", "F32", "F64", "BF16")
