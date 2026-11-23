package runa.platform

import kotlin.test.Test
import kotlin.test.assertTrue

class CpuTest {
    @Test
    fun reportsAPlausibleCoreCount() {
        val cores = computeCoreCount()
        assertTrue(cores in 1..4096, "cores: $cores")
    }
}
