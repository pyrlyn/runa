package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.sizeOf
import platform.windows.GlobalMemoryStatusEx
import platform.windows.MEMORYSTATUSEX

@OptIn(ExperimentalForeignApi::class)
public actual fun physicalMemoryBytes(): Long? =
    memScoped {
        val status = alloc<MEMORYSTATUSEX>()
        status.dwLength = sizeOf<MEMORYSTATUSEX>().convert()
        if (GlobalMemoryStatusEx(status.ptr) != 0) status.ullTotalPhys.toLong() else null
    }
