package runa.engine.llama

import kotlin.test.Test
import kotlin.test.assertEquals

class Utf8AssemblerTest {
    private val assembler = Utf8Assembler()

    @Test
    fun passesWholeTextThrough() {
        assertEquals("héllo", assembler.append("héllo".encodeToByteArray()))
        assertEquals("", assembler.flush())
    }

    @Test
    fun holdsBackASequenceCutByATokenBoundary() {
        val bytes = "a😀b€".encodeToByteArray()
        // Fed one byte at a time, every character must come out once and whole.
        val pieces = bytes.map { assembler.append(byteArrayOf(it)) }
        assertEquals(listOf("a", "", "", "", "😀", "b", "", "", "€"), pieces)
        assertEquals("", assembler.flush())
    }

    @Test
    fun anUnfinishedSequenceBecomesAReplacementCharacterAtTheEnd() {
        assertEquals("x", assembler.append("x😀".encodeToByteArray().copyOf(3)))
        assertEquals("�", assembler.flush())
        assertEquals("", assembler.flush())
    }

    @Test
    fun invalidBytesDoNotStall() {
        // Continuation bytes without a lead byte can never be completed, so they are not held back.
        assertEquals("����", assembler.append(ByteArray(4) { 0x80.toByte() }))
    }
}
