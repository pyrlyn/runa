package runa.gguf

import okio.Buffer
import okio.ByteString
import okio.ByteString.Companion.toByteString
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Property tests of the GGUF reader (PLAN.md, phase 11). A model file comes from the network, so
 * whatever its bytes are, the reader either returns a header or says why it is no GGUF; it never
 * fails another way. Every case comes from its seed, so a failure names the case to run again.
 */
class GgufFuzzTest {
    private class Model(
        val metadata: List<Pair<String, GgufValue>>,
        val tensors: List<GgufTensor>,
        val bytes: ByteString,
    )

    @Test
    fun whatTheWriterWritesIsReadBack() {
        repeat(ROUND_TRIPS) { seed ->
            val model = Random(seed).model()
            val header = readOrReject(model.bytes, "seed $seed") ?: fail("seed $seed: a written header was rejected")
            assertEquals(model.metadata, header.metadata.toList(), "seed $seed")
            assertEquals(model.tensors, header.tensors, "seed $seed")
        }
    }

    // What follows an array is found only when exactly the bytes of the array were passed over.
    @Test
    fun skippingEveryArrayKeepsWhatFollows() {
        repeat(ROUND_TRIPS) { seed ->
            val model = Random(seed).model()
            val header = readOrReject(model.bytes, "seed $seed", maxArrayElements = 0) ?: fail("seed $seed: rejected")
            val skipped =
                model.metadata.map { (key, value) ->
                    key to if (value is GgufValue.Array && value.size > 0) value.copy(elements = null) else value
                }
            assertEquals(skipped, header.metadata.toList(), "seed $seed")
            assertEquals(model.tensors, header.tensors, "seed $seed")
        }
    }

    @Test
    fun damagedHeadersAreReadOrRejected() {
        repeat(MUTATIONS) { seed ->
            val random = Random(seed)
            val damaged = random.damage(random.model().bytes.toByteArray())
            readOrReject(damaged.toByteString(), "seed $seed")
        }
    }

    @Test
    fun arbitraryBytesAreReadOrRejected() {
        repeat(MUTATIONS) { seed ->
            val random = Random(seed)
            val bytes = Buffer()
            if (random.nextInt(8) != 0) bytes.writeUtf8("GGUF").writeIntLe(if (random.nextBoolean()) 3 else 2)
            if (random.nextBoolean()) bytes.writeLongLe(random.nextLong(4)).writeLongLe(random.nextLong(4))
            // Mostly zeros: a length or a count of eight arbitrary bytes is refused at once, and
            // nothing behind it would be read.
            bytes.write(ByteArray(random.nextInt(256)) { if (random.nextInt(4) == 0) random.nextInt().toByte() else 0 })
            readOrReject(bytes.readByteString(), "seed $seed")
        }
    }

    @Test
    fun arraysNestAsDeepAsTheLimitAndNoDeeper() {
        fun nested(depth: Int): GgufValue.Array =
            if (depth == 1) {
                GgufValue.Array(GgufType.UINT8, 1, listOf(GgufValue.Integer(7)))
            } else {
                GgufValue.Array(GgufType.ARRAY, 1, listOf(nested(depth - 1)))
            }

        fun bytes(depth: Int) = Buffer().also(GgufWriter().put("k", GgufType.ARRAY, nested(depth))::writeTo).readByteString()

        assertEquals(nested(4), readOrReject(bytes(4), "four deep")?.metadata?.get("k"))
        assertEquals(null, readOrReject(bytes(5), "five deep"))
    }

    /** The header, or null when the reader refused the bytes; anything else fails the test. */
    private fun readOrReject(
        bytes: ByteString,
        case: String,
        maxArrayElements: Int = DEFAULT_MAX_ARRAY_ELEMENTS,
    ): GgufHeader? {
        val header =
            try {
                readGgufHeader(Buffer().write(bytes), maxArrayElements)
            } catch (e: GgufFormatException) {
                assertTrue(e.message.orEmpty().isNotEmpty(), "$case: rejected without a reason")
                return null
            } catch (e: Throwable) {
                fail("$case: ${e::class.simpleName}: ${e.message} for ${bytes.hex()}")
            }
        // Whatever was read, the answers `runa ls` asks for must not fail either.
        try {
            assertTrue(header.parameterCount >= 0, "$case: ${header.parameterCount} weights for ${bytes.hex()}")
            listOf(header.architecture, header.name, header.type, header.organization, header.quantization, header.chatTemplate)
            listOf(header.contextLength, header.embeddingLength, header.isEmbeddingModel)
        } catch (e: Exception) {
            fail("$case: ${e::class.simpleName}: ${e.message} for ${bytes.hex()}")
        }
        return header
    }

    private fun Random.model(): Model {
        val metadata =
            List(nextInt(8)) { index ->
                val type = GgufType.entries.random(this)
                Triple("${text()}.$index", type, value(type, depth = 1))
            }
        val tensors = List(nextInt(4)) { GgufTensor(text(), List(nextInt(4)) { nextLong(1_000) }, nextInt(40), offset = 0) }
        val writer = GgufWriter(version = if (nextBoolean()) 3 else 2)
        metadata.forEach { (key, type, value) -> writer.put(key, type, value) }
        tensors.forEach { writer.tensor(it.name, *it.dimensions.toLongArray(), type = it.type) }
        return Model(metadata.map { it.first to it.third }, tensors, Buffer().also(writer::writeTo).readByteString())
    }

    private fun Random.value(
        type: GgufType,
        depth: Int,
    ): GgufValue =
        when (type) {
            GgufType.UINT8 -> {
                GgufValue.Integer(nextLong(1L shl 8))
            }

            GgufType.INT8 -> {
                GgufValue.Integer(nextLong(-(1L shl 7), 1L shl 7))
            }

            GgufType.UINT16 -> {
                GgufValue.Integer(nextLong(1L shl 16))
            }

            GgufType.INT16 -> {
                GgufValue.Integer(nextLong(-(1L shl 15), 1L shl 15))
            }

            GgufType.UINT32 -> {
                GgufValue.Integer(nextLong(1L shl 32))
            }

            GgufType.INT32 -> {
                GgufValue.Integer(nextInt().toLong())
            }

            GgufType.UINT64, GgufType.INT64 -> {
                GgufValue.Integer(nextLong())
            }

            // A Float as a Double, so that it is the same number after the file's 32 bits.
            GgufType.FLOAT32 -> {
                GgufValue.Real(listOf(0f, -1.5f, Float.MAX_VALUE, Float.MIN_VALUE, nextFloat()).random(this).toDouble())
            }

            GgufType.FLOAT64 -> {
                GgufValue.Real(listOf(0.0, -1.5, Double.MAX_VALUE, Double.MIN_VALUE, nextDouble()).random(this))
            }

            GgufType.BOOL -> {
                GgufValue.Bool(nextBoolean())
            }

            GgufType.STRING -> {
                GgufValue.Text(text())
            }

            GgufType.ARRAY -> {
                val elements = if (depth == MAX_DEPTH) GgufType.entries - GgufType.ARRAY else GgufType.entries
                val elementType = elements.random(this)
                val size = nextInt(4)
                GgufValue.Array(elementType, size.toLong(), List(size) { value(elementType, depth + 1) })
            }
        }

    // Whole characters only: half of a surrogate pair is not written as it is.
    private fun Random.text(): String = List(nextInt(6)) { LETTERS.random(this) }.joinToString("")

    /** What a broken download, a bad disk or somebody's intent does to a file. */
    private fun Random.damage(bytes: ByteArray): ByteArray {
        var damaged = bytes
        repeat(nextInt(1, 4)) {
            val at = nextInt(damaged.size)
            when (nextInt(5)) {
                0 -> {
                    damaged[at] = nextInt().toByte()
                }

                1 -> {
                    damaged[at] = (damaged[at].toInt() xor (1 shl nextInt(8))).toByte()
                }

                // A count or a length that became one the limits are about.
                2 -> {
                    val number = EDGES.random(this)
                    for (byte in 0 until minOf(Long.SIZE_BYTES, damaged.size - at)) damaged[at + byte] = (number ushr (8 * byte)).toByte()
                }

                3 -> {
                    damaged = damaged.copyOf(maxOf(at, 1))
                }

                // A run written twice: a key that is there again, fields that moved.
                else -> {
                    val run = damaged.copyOfRange(at, minOf(damaged.size, at + nextInt(1, 32)))
                    damaged = damaged.copyOfRange(0, at) + run + damaged.copyOfRange(at, damaged.size)
                }
            }
        }
        return damaged
    }

    private companion object {
        const val ROUND_TRIPS = 500
        const val MUTATIONS = 4_000

        // The reader allows four arrays inside one another; the writer of a real model nests none.
        const val MAX_DEPTH = 4

        val LETTERS = listOf("a", "Z", "0", ".", "_", " ", "é", "世", "🙂", "\u0000", "\n")

        // Each limit of the reader, the number above it, and the ends of what eight bytes hold.
        val EDGES =
            listOf(0, 1, -1, 8, 9, 13, 64, 65, Long.MAX_VALUE, Long.MIN_VALUE) +
                listOf(16, 20, 24, 28, 32).flatMap { listOf(1L shl it, (1L shl it) + 1) }
    }
}
