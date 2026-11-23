package runa.platform

import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class MemoryTest {
    @Test
    fun reportsAPlausibleAmountOfMemory() {
        val bytes = assertNotNull(physicalMemoryBytes())
        // Between 64 MB and 64 TB: wide enough for any machine, tight enough to catch a unit mistake.
        assertTrue(bytes in (1L shl 26)..(1L shl 46), "bytes: $bytes")
    }
}
