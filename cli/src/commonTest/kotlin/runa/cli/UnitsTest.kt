package runa.cli

import runa.cli.output.formatBytes
import runa.cli.output.formatCount
import kotlin.test.Test
import kotlin.test.assertEquals

class UnitsTest {
    @Test
    fun bytesUseDecimalUnitsAndThreeDigits() {
        val cases =
            mapOf(
                0L to "0 B",
                999L to "999 B",
                1_000L to "1.00 KB",
                12_345_678L to "12.3 MB",
                123_456_789L to "123 MB",
                999_499L to "999 KB",
                999_500L to "1.00 MB",
                4_680_000_000L to "4.68 GB",
                2_500_000_000_000_000L to "2500 TB",
            )
        for ((bytes, text) in cases) assertEquals(text, formatBytes(bytes), "bytes: $bytes")
    }

    @Test
    fun countsAreCompact() {
        assertEquals("7.62B", formatCount(7_620_000_000))
        assertEquals("494M", formatCount(494_000_000))
        assertEquals("65.0K", formatCount(65_000))
        assertEquals("12", formatCount(12))
    }
}
