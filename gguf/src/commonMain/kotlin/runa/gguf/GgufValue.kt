package runa.gguf

/** The file is not a GGUF this parser can read: wrong magic, unsupported version, cut short or corrupt. */
public class GgufFormatException(
    message: String,
) : Exception(message)

/** The metadata value types of GGUF, with the ids of `gguf.h` (`enum gguf_type`). */
public enum class GgufType(
    public val id: Int,
    /** Bytes per value; 0 for the variable-size types. */
    internal val byteSize: Int,
) {
    UINT8(0, 1),
    INT8(1, 1),
    UINT16(2, 2),
    INT16(3, 2),
    UINT32(4, 4),
    INT32(5, 4),
    FLOAT32(6, 4),
    BOOL(7, 1),
    STRING(8, 0),
    ARRAY(9, 0),
    UINT64(10, 8),
    INT64(11, 8),
    FLOAT64(12, 8),
    ;

    internal companion object {
        fun of(id: Int): GgufType = entries.firstOrNull { it.id == id } ?: throw GgufFormatException("Unknown GGUF value type $id")
    }
}

public sealed interface GgufValue {
    /** Every integer type. A `UINT64` above `Long.MAX_VALUE` wraps to a negative number. */
    public data class Integer(
        val value: Long,
    ) : GgufValue

    public data class Real(
        val value: Double,
    ) : GgufValue

    public data class Bool(
        val value: Boolean,
    ) : GgufValue

    public data class Text(
        val value: String,
    ) : GgufValue

    /**
     * [elements] is null when the array is longer than the reader's limit: vocabularies have
     * hundreds of thousands of entries and nothing here needs them, so they are skipped on disk.
     */
    public data class Array(
        val elementType: GgufType,
        val size: Long,
        val elements: List<GgufValue>?,
    ) : GgufValue
}

/** A tensor's entry in the header; the weights themselves are never read. */
public data class GgufTensor(
    val name: String,
    val dimensions: List<Long>,
    /** The `ggml_type` id of the weights, for example 0 for F32. */
    val type: Int,
    /** Where the data starts, relative to the start of the tensor data section. */
    val offset: Long,
) {
    val elementCount: Long get() = dimensions.fold(1L) { product, dimension -> product * dimension }
}
