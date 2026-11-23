package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import okio.IOException
import okio.Path
import platform.posix.errno
import platform.posix.strerror

@OptIn(ExperimentalForeignApi::class)
public actual fun createHardLink(
    target: Path,
    link: Path,
) {
    if (platform.posix.link(target.toString(), link.toString()) != 0) {
        throw IOException("cannot link $link to $target: ${strerror(errno)?.toKString()}")
    }
}
