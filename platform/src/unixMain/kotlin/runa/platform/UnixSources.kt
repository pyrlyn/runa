package runa.platform

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix._SC_NPROCESSORS_ONLN
import platform.posix.fgets
import platform.posix.pclose
import platform.posix.popen
import platform.posix.sysconf

internal open class UnixSources : HostSources() {
    override fun processorCount(): Int? = sysconf(_SC_NPROCESSORS_ONLN).toInt().takeIf { it > 0 }

    @OptIn(ExperimentalForeignApi::class)
    override fun output(command: String): String? {
        // The shell reports a missing program on standard error, which nobody asked to see.
        val pipe = popen("$command 2>/dev/null", "r") ?: return null
        val text = StringBuilder()
        memScoped {
            val buffer = allocArray<ByteVar>(BUFFER_BYTES)
            while (fgets(buffer, BUFFER_BYTES, pipe) != null) text.append(buffer.toKString())
        }
        return if (pclose(pipe) == 0) text.toString() else null
    }

    private companion object {
        const val BUFFER_BYTES = 4096
    }
}
