package runa.platform

import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import platform.posix.chmod

public actual fun executablePath(): Path = FileSystem.SYSTEM.canonicalize("/proc/self/exe".toPath())

internal actual fun changeMode(
    path: String,
    mode: Int,
) {
    chmod(path, mode.toUInt())
}
