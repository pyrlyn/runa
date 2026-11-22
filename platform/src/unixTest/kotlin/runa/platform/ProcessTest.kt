package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.value
import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import platform.posix._exit
import platform.posix.fork
import platform.posix.usleep
import platform.posix.waitpid
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ProcessTest {
    private val directory: Path = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-process-test-${Random.nextLong().toULong()}"

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(directory)

    @Test
    fun findsItsOwnExecutable() {
        val path = executablePath()
        assertTrue(path.isAbsolute, "path: $path")
        assertTrue(FileSystem.SYSTEM.exists(path), "path: $path")
    }

    @Test
    fun aLockIsHeldOnceAndFreedByClose() {
        val path = directory / "lock"
        val lock = assertNotNull(FileLock.tryAcquire(path))
        lock.close()
        assertNotNull(FileLock.tryAcquire(path)).close()
    }

    @Test
    fun aDetachedProcessRunsWithItsOutputInAFile() {
        val output = directory / "out.log"
        spawnDetached("/bin/sh".toPath(), listOf("-c", "echo started; echo problem >&2"), output)
        assertEquals(
            "started\nproblem\n",
            waitFor {
                FileSystem.SYSTEM.read(output) { readUtf8() }.takeIf {
                    it.count { c ->
                        c == '\n'
                    } == 2
                }
            },
        )
    }

    @Test
    fun anotherProcessCannotTakeAHeldLock() {
        val path = directory / "lock"
        val lock = assertNotNull(FileLock.tryAcquire(path))
        assertEquals(LOCK_REFUSED, exitCodeOfChild { if (FileLock.tryAcquire(path) == null) LOCK_REFUSED else LOCK_TAKEN })
        lock.close()
        assertEquals(LOCK_TAKEN, exitCodeOfChild { if (FileLock.tryAcquire(path) == null) LOCK_REFUSED else LOCK_TAKEN })
    }

    @Test
    fun anotherProcessSeesWhetherALockIsHeld() {
        val path = directory / "lock"
        assertFalse(FileLock.isHeld(path), "no file yet")
        val lock = assertNotNull(FileLock.tryAcquire(path))
        assertEquals(LOCK_REFUSED, exitCodeOfChild { if (FileLock.isHeld(path)) LOCK_REFUSED else LOCK_TAKEN })
        // Looking must not take the lock away from its owner.
        assertEquals(LOCK_REFUSED, exitCodeOfChild { if (FileLock.tryAcquire(path) == null) LOCK_REFUSED else LOCK_TAKEN })
        lock.close()
        assertEquals(LOCK_TAKEN, exitCodeOfChild { if (FileLock.isHeld(path)) LOCK_REFUSED else LOCK_TAKEN })
    }

    @Test
    fun restrictsAFileToItsOwner() {
        val file = directory / "secret"
        FileSystem.SYSTEM.createDirectories(directory)
        FileSystem.SYSTEM.write(file) { writeUtf8("x") }
        restrictToOwner(file)
        restrictToOwner(directory)
        assertEquals("600", mode(file))
        assertEquals("700", mode(directory))
    }

    private fun mode(path: Path): String {
        val output = directory / "mode-${Random.nextLong().toULong()}"
        // `stat` spells its format flag differently on macOS and Linux; `ls -ld` is the same on both.
        spawnDetached("/bin/sh".toPath(), listOf("-c", "ls -ld '$path' | cut -c2-10"), output)
        val bits = waitFor { FileSystem.SYSTEM.read(output) { readUtf8() }.takeIf { it.endsWith("\n") } }.trim()
        return bits.chunked(3).joinToString("") { triple -> triple.fold(0) { sum, c -> sum * 2 + if (c == '-') 0 else 1 }.toString() }
    }

    @OptIn(ExperimentalForeignApi::class)
    private fun exitCodeOfChild(run: () -> Int): Int {
        val child = fork()
        assertTrue(child >= 0, "fork failed")
        if (child == 0) _exit(run())
        return memScoped {
            val status = alloc<IntVar>()
            assertEquals(child, waitpid(child, status.ptr, 0))
            // WEXITSTATUS is a C macro: the exit code is the second byte.
            (status.value shr 8) and 0xff
        }
    }

    private fun <T : Any> waitFor(read: () -> T?): T {
        repeat(200) {
            runCatching(read).getOrNull()?.let { return it }
            usleep(25_000u)
        }
        error("timed out")
    }

    private companion object {
        const val LOCK_REFUSED = 7
        const val LOCK_TAKEN = 8
    }
}
