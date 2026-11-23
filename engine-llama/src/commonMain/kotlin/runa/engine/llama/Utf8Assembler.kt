package runa.engine.llama

/**
 * Turns token pieces into text. A token is a run of bytes and may end in the middle of a UTF-8
 * sequence (an emoji is often several tokens), so bytes are held back until the sequence is whole.
 */
internal class Utf8Assembler {
    private var pending = ByteArray(0)

    /** The text that [bytes] completes; empty while a sequence is still open. */
    fun append(bytes: ByteArray): String {
        val all = pending + bytes
        val complete = completePrefix(all)
        pending = all.copyOfRange(complete, all.size)
        return all.decodeToString(0, complete)
    }

    /** What is left at the end of generation; an unfinished sequence becomes U+FFFD. */
    fun flush(): String = pending.decodeToString().also { pending = ByteArray(0) }

    private fun completePrefix(bytes: ByteArray): Int {
        // Only the last sequence can be open, and its lead byte is at most 3 bytes from the end.
        for (back in 1..minOf(MAX_SEQUENCE - 1, bytes.size)) {
            val byte = bytes[bytes.size - back].toInt() and 0xFF
            if (byte and 0xC0 == 0x80) continue
            return if (sequenceLength(byte) > back) bytes.size - back else bytes.size
        }
        return bytes.size
    }

    private fun sequenceLength(lead: Int): Int =
        when {
            lead and 0xE0 == 0xC0 -> 2
            lead and 0xF0 == 0xE0 -> 3
            lead and 0xF8 == 0xF0 -> 4
            else -> 1
        }

    private companion object {
        const val MAX_SEQUENCE = 4
    }
}
