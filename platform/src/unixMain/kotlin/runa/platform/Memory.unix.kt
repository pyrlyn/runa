package runa.platform

import platform.posix._SC_PAGESIZE
import platform.posix._SC_PHYS_PAGES
import platform.posix.sysconf

public actual fun physicalMemoryBytes(): Long? {
    val pages = sysconf(_SC_PHYS_PAGES)
    val pageSize = sysconf(_SC_PAGESIZE)
    return if (pages > 0 && pageSize > 0) pages * pageSize else null
}
