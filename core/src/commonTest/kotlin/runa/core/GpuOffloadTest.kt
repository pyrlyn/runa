package runa.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class GpuOffloadTest {
    @Test
    fun parsesOffMaxAndRatios() {
        assertEquals(GpuOffload.Off, GpuOffload.parse("off"))
        assertEquals(GpuOffload.Max, GpuOffload.parse("max"))
        assertEquals(GpuOffload.Ratio(0.5), GpuOffload.parse("0.5"))
        assertEquals(GpuOffload.Ratio(0.0), GpuOffload.parse("0"))
        assertEquals(GpuOffload.Ratio(1.0), GpuOffload.parse("1"))
    }

    @Test
    fun rejectsEverythingElse() {
        for (text in listOf("", "auto", "1.5", "-0.1", "NaN", "MAX")) {
            assertNull(GpuOffload.parse(text), "input: '$text'")
        }
    }

    @Test
    fun idRoundTrips() {
        for (offload in listOf(GpuOffload.Off, GpuOffload.Max, GpuOffload.Ratio(0.25))) {
            assertEquals(offload, GpuOffload.parse(offload.id))
        }
    }
}
