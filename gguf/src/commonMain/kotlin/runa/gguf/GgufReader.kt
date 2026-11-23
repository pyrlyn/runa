package runa.gguf

import okio.BufferedSource
import okio.EOFException
import okio.FileSystem
import okio.Path
import okio.buffer
import okio.use

/**
 * Reads the header of the GGUF file at [path]. Only the header is read, in order, so the cost
 * does not depend on the size of the weights.
 *
 * @throws GgufFormatException when the file is not a readable GGUF.
 * @throws okio.IOException when the file cannot be read at all.
 */
public fun readGgufHeader(
    fileSystem: FileSystem,
    path: Path,
    maxArrayElements: Int = DEFAULT_MAX_ARRAY_ELEMENTS,
): GgufHeader = fileSystem.source(path).buffer().use { readGgufHeader(it, maxArrayElements) }

/**
 * Reads a GGUF header from [source]. Only top-level arrays of at most [maxArrayElements] elements
 * are kept, and only while the header's budget of [MAX_KEPT_ELEMENTS] kept elements lasts; every
 * other array is skipped on disk and comes back with `elements = null`.
 */
public fun readGgufHeader(
    source: BufferedSource,
    maxArrayElements: Int = DEFAULT_MAX_ARRAY_ELEMENTS,
): GgufHeader =
    try {
        Reader(source, maxArrayElements).header()
    } catch (e: EOFException) {
        throw GgufFormatException("The file ends inside its GGUF header")
    }

/** Long enough for every scalar-like array (dimensions, shapes), far below a vocabulary. */
public const val DEFAULT_MAX_ARRAY_ELEMENTS: Int = 64

/**
 * How many array elements one header may keep in total, across all of its metadata entries. The
 * per-array limit alone does not bound memory: nested arrays multiply it per level, and a header
 * may carry tens of thousands of entries. Real models keep a few dozen elements at most.
 */
public const val MAX_KEPT_ELEMENTS: Long = 1L shl 12

// The limits below reject corrupt or hostile headers before they cost memory or time. Real models
// are orders of magnitude under each of them.
private const val MAX_METADATA_ENTRIES = 1L shl 16
private const val MAX_TENSORS = 1L shl 20
private const val MAX_KEY_BYTES = 1L shl 16
private const val MAX_STRING_BYTES = 1L shl 24
private const val MAX_ARRAY_ELEMENTS = 1L shl 28
private const val MAX_ARRAY_DEPTH = 4
private const val MAX_DIMENSIONS = 8

// "GGUF" read as a little-endian 32-bit integer.
private const val MAGIC = 0x46554747

private class Reader(
    private val source: BufferedSource,
    private val maxArrayElements: Int,
) {
    // What is left of MAX_KEPT_ELEMENTS for this header; charged before elements are read.
    private var keptLeft: Long = MAX_KEPT_ELEMENTS

    fun header(): GgufHeader {
        if (!source.request(4) || source.readIntLe() != MAGIC) throw GgufFormatException("Not a GGUF file")
        val version = source.readIntLe()
        when {
            version == 2 || version == 3 -> Unit

            // A small version number lands in the high bytes when the file was written big-endian.
            version and 0xFFFF == 0 -> throw GgufFormatException("Big-endian GGUF files are not supported")

            else -> throw GgufFormatException("GGUF version $version is not supported")
        }
        val tensorCount = count("tensor count", MAX_TENSORS)
        val metadataCount = count("metadata count", MAX_METADATA_ENTRIES)

        val metadata = LinkedHashMap<String, GgufValue>()
        repeat(metadataCount) {
            val key = string(MAX_KEY_BYTES)
            val value = value(GgufType.of(source.readIntLe()), depth = 0)
            if (metadata.put(key, value) != null) throw GgufFormatException("Duplicate metadata key $key")
        }
        var weights = 0L
        val tensors =
            List(tensorCount) {
                tensor().also {
                    // Each count fits a Long, and their sum must too: it is the parameter count.
                    if (weights >
                        Long.MAX_VALUE - it.elementCount
                    ) {
                        throw GgufFormatException("The tensors hold an implausible number of weights")
                    }
                    weights += it.elementCount
                }
            }
        return GgufHeader(version, metadata, tensors)
    }

    private fun count(
        what: String,
        max: Long,
    ): Int {
        val value = source.readLongLe()
        if (value !in 0..max) throw GgufFormatException("Implausible $what: ${value.toULong()}")
        return value.toInt()
    }

    private fun string(maxBytes: Long = MAX_STRING_BYTES): String {
        val length = source.readLongLe()
        if (length !in 0..maxBytes) throw GgufFormatException("Implausible string length: ${length.toULong()}")
        return source.readUtf8(length)
    }

    private fun value(
        type: GgufType,
        depth: Int,
    ): GgufValue =
        when (type) {
            GgufType.UINT8 -> GgufValue.Integer(source.readByte().toLong() and 0xFF)
            GgufType.INT8 -> GgufValue.Integer(source.readByte().toLong())
            GgufType.UINT16 -> GgufValue.Integer(source.readShortLe().toLong() and 0xFFFF)
            GgufType.INT16 -> GgufValue.Integer(source.readShortLe().toLong())
            GgufType.UINT32 -> GgufValue.Integer(source.readIntLe().toLong() and 0xFFFFFFFFL)
            GgufType.INT32 -> GgufValue.Integer(source.readIntLe().toLong())
            GgufType.UINT64, GgufType.INT64 -> GgufValue.Integer(source.readLongLe())
            GgufType.FLOAT32 -> GgufValue.Real(Float.fromBits(source.readIntLe()).toDouble())
            GgufType.FLOAT64 -> GgufValue.Real(Double.fromBits(source.readLongLe()))
            GgufType.BOOL -> GgufValue.Bool(source.readByte() != 0.toByte())
            GgufType.STRING -> GgufValue.Text(string())
            GgufType.ARRAY -> array(depth)
        }

    private fun array(depth: Int): GgufValue.Array {
        if (depth == MAX_ARRAY_DEPTH) throw GgufFormatException("Arrays are nested too deeply")
        val elementType = GgufType.of(source.readIntLe())
        val size = source.readLongLe()
        if (size !in 0..MAX_ARRAY_ELEMENTS) throw GgufFormatException("Implausible array length: ${size.toULong()}")
        // Only top-level arrays are kept: nothing reads a nested one, and keeping them would let one
        // entry multiply the per-array limit by itself once per level of nesting.
        if (depth == 0 && size <= maxArrayElements && size <= keptLeft) {
            keptLeft -= size
            return GgufValue.Array(elementType, size, List(size.toInt()) { value(elementType, depth + 1) })
        }
        when {
            elementType.byteSize > 0 -> source.skip(size * elementType.byteSize)

            elementType == GgufType.STRING -> repeat(size.toInt()) { skipString() }

            // Nested arrays are walked, not kept; MAX_ARRAY_DEPTH bounds the recursion, the file size the work.
            else -> repeat(size.toInt()) { array(depth + 1) }
        }
        return GgufValue.Array(elementType, size, elements = null)
    }

    private fun skipString() {
        val length = source.readLongLe()
        if (length !in 0..MAX_STRING_BYTES) throw GgufFormatException("Implausible string length: ${length.toULong()}")
        source.skip(length)
    }

    private fun tensor(): GgufTensor {
        val name = string(MAX_KEY_BYTES)
        val dimensionCount = source.readIntLe()
        if (dimensionCount !in 0..MAX_DIMENSIONS) throw GgufFormatException("Tensor $name has $dimensionCount dimensions")
        var elements = 1L
        val dimensions =
            List(dimensionCount) {
                val dimension = source.readLongLe()
                // The product must stay a valid Long: parameter counts are sums of these.
                if (dimension < 0 || (dimension != 0L && elements > Long.MAX_VALUE / dimension)) {
                    throw GgufFormatException("Tensor $name has an implausible shape")
                }
                elements *= dimension
                dimension
            }
        val type = source.readIntLe()
        val offset = source.readLongLe()
        return GgufTensor(name, dimensions, type, offset)
    }
}
