package runa.gguf

import okio.BufferedSink

/**
 * Writes a GGUF header with no tensor data behind it. That is all a model file is to the store
 * and to `runa ls`, so tests and fixtures are built with this instead of keeping weights in git.
 */
public class GgufWriter(
    private val version: Int = 3,
) {
    private val metadata = mutableListOf<Triple<String, GgufType, GgufValue>>()
    private val tensors = mutableListOf<GgufTensor>()

    public fun put(
        key: String,
        type: GgufType,
        value: GgufValue,
    ): GgufWriter = apply { metadata += Triple(key, type, value) }

    public fun put(
        key: String,
        value: String,
    ): GgufWriter = put(key, GgufType.STRING, GgufValue.Text(value))

    public fun put(
        key: String,
        value: Long,
        type: GgufType = GgufType.UINT32,
    ): GgufWriter = put(key, type, GgufValue.Integer(value))

    public fun tensor(
        name: String,
        vararg dimensions: Long,
        type: Int = 0,
    ): GgufWriter = apply { tensors += GgufTensor(name, dimensions.toList(), type, offset = 0) }

    public fun writeTo(sink: BufferedSink) {
        sink.writeUtf8("GGUF").writeIntLe(version)
        sink.writeLongLe(tensors.size.toLong()).writeLongLe(metadata.size.toLong())
        for ((key, type, value) in metadata) {
            sink.string(key).writeIntLe(type.id)
            sink.value(type, value)
        }
        for (tensor in tensors) {
            sink.string(tensor.name).writeIntLe(tensor.dimensions.size)
            tensor.dimensions.forEach(sink::writeLongLe)
            sink.writeIntLe(tensor.type).writeLongLe(tensor.offset)
        }
    }

    private fun BufferedSink.string(text: String): BufferedSink {
        val bytes = text.encodeToByteArray()
        return writeLongLe(bytes.size.toLong()).write(bytes)
    }

    private fun BufferedSink.value(
        type: GgufType,
        value: GgufValue,
    ) {
        when (type) {
            GgufType.UINT8, GgufType.INT8 -> {
                writeByte(value.integer().toInt())
            }

            GgufType.UINT16, GgufType.INT16 -> {
                writeShortLe(value.integer().toInt())
            }

            GgufType.UINT32, GgufType.INT32 -> {
                writeIntLe(value.integer().toInt())
            }

            GgufType.UINT64, GgufType.INT64 -> {
                writeLongLe(value.integer())
            }

            GgufType.FLOAT32 -> {
                writeIntLe((value as GgufValue.Real).value.toFloat().toRawBits())
            }

            GgufType.FLOAT64 -> {
                writeLongLe((value as GgufValue.Real).value.toRawBits())
            }

            GgufType.BOOL -> {
                writeByte(if ((value as GgufValue.Bool).value) 1 else 0)
            }

            GgufType.STRING -> {
                string((value as GgufValue.Text).value)
            }

            GgufType.ARRAY -> {
                val array = value as GgufValue.Array
                val elements = requireNotNull(array.elements) { "Cannot write an array whose elements were skipped" }
                writeIntLe(array.elementType.id).writeLongLe(elements.size.toLong())
                elements.forEach { value(array.elementType, it) }
            }
        }
    }

    private fun GgufValue.integer(): Long = (this as GgufValue.Integer).value
}
