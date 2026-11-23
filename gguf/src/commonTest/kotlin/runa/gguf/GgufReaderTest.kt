package runa.gguf

import okio.Buffer
import okio.ByteString
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

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
        // A nested array is passed over, not kept: its type and size are read, its elements are not.
        val expected =
            values.map { it.second }.dropLast(1) +
                GgufValue.Array(GgufType.ARRAY, 1, listOf(GgufValue.Array(GgufType.STRING, 1, null)))
        assertEquals(
            expected,
            writer
                .read()
                .metadata.values
                .toList(),
        )
    }

    @Test
    fun aNestedArrayBombIsSkippedNotKept() {
        // 64^4 values in one entry: each level is within the per-array limit. The levels share one
        // child, so the test itself does not hold 16 million values.
        fun bomb(depth: Int): GgufValue.Array =
            if (depth == 1) {
                GgufValue.Array(GgufType.UINT8, 64, List(64) { GgufValue.Integer(7) })
            } else {
                val child = bomb(depth - 1)
                GgufValue.Array(GgufType.ARRAY, 64, List(64) { child })
            }

        val bytes =
            GgufWriter()
                .put("bomb", GgufType.ARRAY, bomb(4))
                .put("general.architecture", "llama")
                .bytes()
        val started = TimeSource.Monotonic.markNow()
        val header = readGgufHeader(Buffer().write(bytes))
        val elapsed = started.elapsedNow()

        val value = assertIs<GgufValue.Array>(header.metadata["bomb"])
        assertEquals(GgufType.ARRAY, value.elementType)
        assertEquals(64, value.size)
        assertTrue(header.metadata.values.sumOf(::kept) <= MAX_KEPT_ELEMENTS, "kept ${header.metadata.values.sumOf(::kept)}")
        assertEquals("llama", header.architecture)
        // Generous for slow CI; keeping every value took seconds and a gigabyte.
        assertTrue(elapsed < 5.seconds, "read took $elapsed")
    }

    @Test
    fun manySmallArraysShareOneBudget() {
        val small = GgufValue.Array(GgufType.UINT8, 64, List(64) { GgufValue.Integer(1) })
        val writer = GgufWriter()
        repeat(1_000) { writer.put("a$it", GgufType.ARRAY, small) }
        val header = writer.put("general.architecture", "llama").read()

        assertEquals(1_001, header.metadata.size)
        val arrays = (0 until 1_000).map { assertIs<GgufValue.Array>(header.metadata["a$it"], "a$it") }
        assertTrue(arrays.all { it.size == 64L && it.elementType == GgufType.UINT8 })
        val keptCount = (MAX_KEPT_ELEMENTS / 64).toInt()
        assertTrue(arrays.take(keptCount).all { it.elements == small.elements })
        assertTrue(arrays.drop(keptCount).all { it.elements == null })
        assertEquals(MAX_KEPT_ELEMENTS, header.metadata.values.sumOf(::kept))
        assertEquals("llama", header.architecture)
    }

    /** How many array elements [value] holds in memory, at every level. */
    private fun kept(value: GgufValue): Long = if (value is GgufValue.Array) value.elements.orEmpty().sumOf { 1 + kept(it) } else 0

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

        val overflowingSum = header(2, 0)
        repeat(2) {
            overflowingSum
                .writeLongLe(1)
                .writeUtf8("t")
                .writeIntLe(1)
                .writeLongLe(Long.MAX_VALUE)
                .writeIntLe(0)
                .writeLongLe(0)
        }
        assertEquals("The tensors hold an implausible number of weights", failure(overflowingSum.readByteString()))
    }

    @Test
    fun duplicateKeysAreRejected() {
        assertEquals("Duplicate metadata key a", failure(GgufWriter().put("a", 1).put("a", 2).bytes()))
    }
}
