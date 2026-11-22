package runa.platform

import okio.FileSystem
import okio.IOException
import okio.Path.Companion.toPath
import platform.posix._SC_NPROCESSORS_ONLN
import platform.posix.sysconf

public actual fun computeCoreCount(): Int = physicalCores() ?: sysconf(_SC_NPROCESSORS_ONLN).toInt().coerceAtLeast(1)

/** Hyper-threads of one core list the same siblings, so the distinct lists are the cores. */
private fun physicalCores(): Int? =
    try {
        FileSystem.SYSTEM
            .list("/sys/devices/system/cpu".toPath())
            .filter { it.name.startsWith("cpu") && it.name.drop(3).toIntOrNull() != null }
            .mapNotNull { cpu ->
                val siblings = cpu / "topology" / "thread_siblings_list"
                if (FileSystem.SYSTEM.exists(siblings)) FileSystem.SYSTEM.read(siblings) { readUtf8() }.trim() else null
            }.distinct()
            .size
            .takeIf { it > 0 }
    } catch (e: IOException) {
        null
    }
