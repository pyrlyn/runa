package runa.daemon

import okio.FileSystem
import okio.Path
import okio.buffer
import okio.use
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals

class LogFileTest {
    private val fileSystem = FileSystem.SYSTEM
    private val directory = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-log-test-${Random.nextLong().toULong()}"
    private val path = directory / "daemon.log"
    private val reopened = mutableListOf<Path>()

    @BeforeTest
    fun setUp() = fileSystem.createDirectories(directory)

    @AfterTest
    fun cleanUp() = fileSystem.deleteRecursively(directory)

    /** A log whose output is the file itself, as the daemon's is. */
    private fun logFile(maxBytes: Long) =
        LogFile(
            path,
            fileSystem,
            maxBytes,
            kept = 2,
            print = { line -> fileSystem.appendingSink(path).buffer().use { it.writeUtf8(line + "\n") } },
            reopen = { reopened += it },
        )

    private fun files() = fileSystem.list(directory).map { it.name }.sorted()

    private fun read(name: String) = fileSystem.read(directory / name) { readUtf8() }

    @Test
    fun aFullFileIsMovedAsideAndTheOldestIsDropped() {
        val log = logFile(maxBytes = 10)
        log.write("first")
        assertEquals(listOf("daemon.log"), files())
        assertEquals(emptyList(), reopened)

        log.write("second")
        assertEquals(listOf("daemon.log.1"), files())
        assertEquals("first\nsecond\n", read("daemon.log.1"))
        assertEquals(listOf(path), reopened)

        log.write("a third line")
        log.write("a fourth line")
        // Two old files are kept, so the first is gone.
        assertEquals(listOf("daemon.log.1", "daemon.log.2"), files())
        assertEquals("a fourth line\n", read("daemon.log.1"))
        assertEquals("a third line\n", read("daemon.log.2"))
        assertEquals(3, reopened.size)
    }

    @Test
    fun whatTheFileHeldBeforeCounts() {
        fileSystem.write(path) { writeUtf8("x".repeat(95)) }
        val log = logFile(maxBytes = 100)
        log.write("more")
        assertEquals(listOf("daemon.log.1"), files())
    }

    @Test
    fun aFileThatCannotBeMovedIsReportedAndTheLogGoesOn() {
        val failures = mutableListOf<String>()
        val written = mutableListOf<String>()
        // Nothing reaches the file, so there is none to move.
        val log = LogFile(path, fileSystem, maxBytes = 1, print = written::add, reopen = { reopened += it })
        log.onFailure = failures::add
        log.write("one")
        log.write("two")
        assertEquals(listOf("one", "two"), written)
        assertEquals(2, failures.size)
        assertContains(failures.first(), "Cannot rotate $path")
        assertEquals(emptyList(), reopened)
    }
}
