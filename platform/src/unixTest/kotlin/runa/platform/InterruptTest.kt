package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.value
import platform.posix.SIGINT
import platform.posix._exit
import platform.posix.fork
import platform.posix.getpid
import platform.posix.kill
import platform.posix.waitpid
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@OptIn(ExperimentalForeignApi::class)
class InterruptTest {
    @Test
    fun interruptExitsWithTheGivenCode() {
        val child = fork()
        assertTrue(child >= 0, "fork failed")
        if (child == 0) {
            exitOnInterrupt(130)
            kill(getpid(), SIGINT)
            // Only reached when the handler did not run.
            _exit(99)
        }
        memScoped {
            val status = alloc<kotlinx.cinterop.IntVar>()
            assertEquals(child, waitpid(child, status.ptr, 0))
            // WIFEXITED and WEXITSTATUS are C macros: a normal exit has a zero low byte and the
            // code in the next one.
            assertEquals(0, status.value and 0x7f, "child was killed by a signal")
            assertEquals(130, (status.value shr 8) and 0xff)
        }
    }
}
