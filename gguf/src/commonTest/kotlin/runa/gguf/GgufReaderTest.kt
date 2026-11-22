package runa.gguf

import okio.Buffer
import okio.ByteString
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GgufReaderTest {
    private fun GgufWriter.bytes(): ByteString = Buffer().also(::writeTo).readByteString()

    private fun GgufWriter.read(maxArrayElements: Int = DEFAULT_MAX_ARRAY_ELEMENTS) =
        readGgufHeader(Buffer().write(bytes()), maxArrayElements)

    private fun failure(bytes: ByteString): String =
        assertFailsWith<GgufFormatException> { readGgufHeader(Buffer().write(bytes)) }.message.orEmpty()

    private fun llm() =
        GgufWriter()
            .put("general.architecture", "llama")
            .put("general.name", "Tiny Llama")
            .put("general.file_type", 15)
            .put("llama.context_length", 4096)
            .put("llama.embedding_length", 64)
            .put("tokenizer.chat_template", "{{ messages }}")
            .tensor("token_embd.weight", 64, 1000)
            .tensor("output.bias", 1000, type = 1)

    @Test
    fun readsTheFieldsOfALanguageModel() {
        val header = llm().read()
        assertEquals(3, header.version)
        assertEquals("llama", header.architecture)
        assertEquals("Tiny Llama", header.name)
        assertEquals("Q4_K_M", header.quantization)
        assertEquals(4096, header.contextLength)
        assertEquals(64, header.embeddingLength)
        assertEquals("{{ messages }}", header.chatTemplate)
        assertEquals(65_000, header.parameterCount)
        assertFalse(header.isEmbeddingModel)
        assertEquals(
            listOf(GgufTensor("token_embd.weight", listOf(64, 1000), 0, 0), GgufTensor("output.bias", listOf(1000), 1, 0)),
            header.tensors,
        )
    }

    @Test
    fun readsEveryValueType() {
        val values =
            listOf(
                GgufType.UINT8 to GgufValue.Integer(255),
                GgufType.INT8 to GgufValue.Integer(-128),
                GgufType.UINT16 to GgufValue.Integer(65_535),
                GgufType.INT16 to GgufValue.Integer(-32_768),
                GgufType.UINT32 to GgufValue.Integer(4_294_967_295),
                GgufType.INT32 to GgufValue.Integer(-2_147_483_648),
                GgufType.UINT64 to GgufValue.Integer(Long.MAX_VALUE),
                GgufType.INT64 to GgufValue.Integer(Long.MIN_VALUE),
                GgufType.FLOAT32 to GgufValue.Real(0.5),
                GgufType.FLOAT64 to GgufValue.Real(1e300),
                GgufType.BOOL to GgufValue.Bool(true),
                GgufType.STRING to GgufValue.Text("héllo, 世界"),
                GgufType.ARRAY to GgufValue.Array(GgufType.INT32, 2, listOf(GgufValue.Integer(-1), GgufValue.Integer(7))),
                GgufType.ARRAY to
                    GgufValue.Array(
                        GgufType.ARRAY,
                        1,
                        listOf(GgufValue.Array(GgufType.STRING, 1, listOf(GgufValue.Text("nested")))),
                    ),
            )
        val writer = GgufWriter()
        values.forEachIndexed { index, (type, value) -> writer.put("k$index", type, value) }
        assertEquals(
            values.map { it.second },
            writer
                .read()
                .metadata.values
                .toList(),
        )
    }

    @Test
    fun longArraysAreSkippedButWhatFollowsIsStillRead() {
        fun array(
            type: GgufType,
            element: GgufValue,
        ) = GgufValue.Array(type, 3, List(3) { element })

        val header =
            GgufWriter()
                .put("tokens", GgufType.ARRAY, array(GgufType.STRING, GgufValue.Text("tok")))
                .put("scores", GgufType.ARRAY, array(GgufType.FLOAT32, GgufValue.Real(1.0)))
                .put("nested", GgufType.ARRAY, array(GgufType.ARRAY, GgufValue.Array(GgufType.UINT8, 1, listOf(GgufValue.Integer(1)))))
                .put("general.architecture", "llama")
                .tensor("t", 2, 3)
                .read(maxArrayElements = 2)
        assertEquals(GgufValue.Array(GgufType.STRING, 3, null), header.metadata["tokens"])
        assertEquals(GgufValue.Array(GgufType.FLOAT32, 3, null), header.metadata["scores"])
        assertEquals(GgufValue.Array(GgufType.ARRAY, 3, null), header.metadata["nested"])
        assertEquals("llama", header.architecture)
        assertEquals(6, header.parameterCount)
    }

    @Test
    fun version2IsRead() {
        assertEquals(2, GgufWriter(version = 2).put("general.architecture", "llama").read().version)
    }

    @Test
    fun embeddingModelsAreRecognised() {
        assertTrue(GgufWriter().put("general.architecture", "nomic-bert").read().isEmbeddingModel)
        assertTrue(
            GgufWriter()
                .put("general.architecture", "gemma-embedding")
                .put("gemma-embedding.pooling_type", 1)
                .read()
                .isEmbeddingModel,
        )
        assertFalse(
            GgufWriter()
                .put("general.architecture", "llama")
                .put("llama.pooling_type", 0)
                .read()
                .isEmbeddingModel,
        )
    }

    @Test
    fun quantizationIgnoresTheGuessedFlagAndUnknownIds() {
        assertEquals("Q8_0", GgufWriter().put("general.file_type", 1024 + 7).read().quantization)
        assertNull(GgufWriter().put("general.file_type", 999).read().quantization)
        assertNull(GgufWriter().read().quantization)
    }

    @Test
    fun aFileCutAnywhereInsideTheHeaderIsRejected() {
        val bytes = llm().bytes()
        for (length in 0 until bytes.size) {
            val message = failure(bytes.substring(0, length))
            assertTrue(message == "Not a GGUF file" || message == "The file ends inside its GGUF header", "cut at $length: $message")
        }
    }

    @Test
    fun otherFilesAreRejected() {
        assertEquals("Not a GGUF file", failure(Buffer().writeUtf8("PK\u0003\u0004 a zip archive").readByteString()))
        assertEquals("Not a GGUF file", failure(ByteString.EMPTY))
        assertEquals("GGUF version 1 is not supported", failure(GgufWriter(version = 1).bytes()))
        assertEquals("Big-endian GGUF files are not supported", failure(Buffer().writeUtf8("GGUF").writeInt(3).readByteString()))
    }

    @Test
    fun implausibleCountsAndLengthsAreRejectedWithoutAllocating() {
        fun header(
            tensors: Long,
            entries: Long,
        ) = Buffer()
            .writeUtf8("GGUF")
            .writeIntLe(3)
            .writeLongLe(tensors)
            .writeLongLe(entries)

        assertContains(failure(header(tensors = -1, entries = 0).readByteString()), "tensor count")
        assertContains(failure(header(tensors = 0, entries = 1L shl 40).readByteString()), "metadata count")
        assertContains(failure(header(0, 1).writeLongLe(1L shl 40).readByteString()), "string length")

        val hugeArray = header(0, 1).writeLongLe(1).writeUtf8("k").writeIntLe(GgufType.ARRAY.id)
        hugeArray.writeIntLe(GgufType.UINT8.id).writeLongLe(1L shl 40)
        assertContains(failure(hugeArray.readByteString()), "array length")

        val unknownType = header(0, 1).writeLongLe(1).writeUtf8("k").writeIntLe(99)
        assertEquals("Unknown GGUF value type 99", failure(unknownType.readByteString()))

        val overflowingShape = header(1, 0).writeLongLe(1).writeUtf8("t").writeIntLe(2)
        overflowingShape
            .writeLongLe(Long.MAX_VALUE)
            .writeLongLe(4)
            .writeIntLe(0)
            .writeLongLe(0)
        assertEquals("Tensor t has an implausible shape", failure(overflowingShape.readByteString()))
    }

    @Test
    fun duplicateKeysAreRejected() {
        assertEquals("Duplicate metadata key a", failure(GgufWriter().put("a", 1).put("a", 2).bytes()))
    }
}
