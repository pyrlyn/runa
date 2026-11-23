package runa.engine.llama

/**
 * Cuts generated text at the first stop string. A stop string can arrive split over several
 * tokens, so text that may be the start of one is held back until the next token settles it.
 */
internal class StopFilter(
    stops: List<String>,
) {
    private val stops = stops.filter { it.isNotEmpty() }
    private val held = StringBuilder()

    /** True once a stop string was found; nothing more is emitted after that. */
    var stopped: Boolean = false
        private set

    /** The part of the text so far that is safe to show. */
    fun append(text: String): String {
        if (stopped) return ""
        held.append(text)
        val cut = stops.map { held.indexOf(it) }.filter { it >= 0 }.minOrNull()
        if (cut != null) {
            stopped = true
            return take(cut)
        }
        val keep = stops.maxOfOrNull(::openPrefix) ?: 0
        return take(held.length - keep)
    }

    /** The held-back text, once generation ended without a stop string. */
    fun flush(): String = if (stopped) "" else take(held.length)

    /** The length of the longest end of the held text that begins [stop] without completing it. */
    private fun openPrefix(stop: String): Int =
        (minOf(stop.length - 1, held.length) downTo 1).firstOrNull { held.endsWith(stop.substring(0, it)) } ?: 0

    private fun take(length: Int): String {
        val text = held.substring(0, length)
        if (stopped) held.clear() else held.deleteRange(0, length)
        return text
    }
}
