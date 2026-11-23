package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.sizeOf
import kotlinx.cinterop.value
import platform.darwin.sysctlbyname
import platform.posix.size_tVar

// perflevel0 is the performance cluster; Intel Macs do not have the key.
public actual fun computeCoreCount(): Int = sysctlInt("hw.perflevel0.physicalcpu") ?: sysctlInt("hw.physicalcpu") ?: 1

@OptIn(ExperimentalForeignApi::class)
private fun sysctlInt(name: String): Int? =
    memScoped {
        val value = alloc<IntVar>()
        val size = alloc<size_tVar>()
        size.value = sizeOf<IntVar>().convert()
        if (sysctlbyname(name, value.ptr, size.ptr, null, 0u) == 0 && value.value > 0) value.value else null
    }
