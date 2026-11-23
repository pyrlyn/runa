package runa.engine.llama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class StopFilterTest {
    @Test
    fun withoutStopStringsNothingIsHeldBack() {
        val filter = StopFilter(listOf(""))
        assertEquals("abc", filter.append("abc"))
        assertEquals("", filter.flush())
        assertFalse(filter.stopped)
    }

    @Test
    fun cutsAtAStopStringInsideOnePiece() {
        val filter = StopFilter(listOf("END"))
        assertEquals("a b ", filter.append("a b END c"))
        assertTrue(filter.stopped)
        assertEquals("", filter.append("more"))
        assertEquals("", filter.flush())
    }

    @Test
    fun findsAStopStringSplitOverPieces() {
        val filter = StopFilter(listOf("</s>"))
        assertEquals("hi", filter.append("hi<"))
        assertEquals("", filter.append("/s"))
        assertEquals("", filter.append(">tail"))
        assertTrue(filter.stopped)
    }

    @Test
    fun releasesTextThatOnlyLookedLikeAStopString() {
        val filter = StopFilter(listOf("</s>"))
        assertEquals("a", filter.append("a</"))
        assertEquals("</b", filter.append("b"))
        assertEquals("", filter.append("<"))
        assertEquals("<", filter.flush())
        assertFalse(filter.stopped)
    }

    @Test
    fun theEarliestStopStringWins() {
        val filter = StopFilter(listOf("world", "lo"))
        assertEquals("hel", filter.append("hello world"))
        assertTrue(filter.stopped)
    }
}
