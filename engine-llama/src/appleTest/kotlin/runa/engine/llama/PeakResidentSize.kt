package runa.engine.llama

import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import platform.posix.RUSAGE_SELF
import platform.posix.getrusage
import platform.posix.rusage

internal actual fun peakResidentSize(): Long =
    memScoped {
        val usage = alloc<rusage>()
        check(getrusage(RUSAGE_SELF, usage.ptr) == 0) { "getrusage failed" }
        usage.ru_maxrss
    }
