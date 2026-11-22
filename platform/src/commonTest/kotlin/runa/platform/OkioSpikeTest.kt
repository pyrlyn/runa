package runa.platform

import okio.FileSystem
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

// Phase 0 spike (PLAN.md): Okio's system file system writes and reads files on native.
class OkioSpikeTest {
    @Test
    fun writesAndReadsFile() {
        val fs = FileSystem.SYSTEM
        val dir = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-okio-spike-${Random.nextLong().toULong()}"
        fs.createDirectories(dir)
        try {
            val file = dir / "hello.txt"
            fs.write(file) { writeUtf8("hello") }
            assertEquals("hello", fs.read(file) { readUtf8() })
        } finally {
            fs.deleteRecursively(dir)
        }
    }
}
