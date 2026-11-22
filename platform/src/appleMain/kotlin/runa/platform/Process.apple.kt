package runa.platform

import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import platform.Foundation.NSBundle
import platform.posix.chmod

public actual fun executablePath(): Path =
    FileSystem.SYSTEM.canonicalize(checkNotNull(NSBundle.mainBundle.executablePath) { "the process has no executable path" }.toPath())

internal actual fun changeMode(
    path: String,
    mode: Int,
) {
    chmod(path, mode.toUShort())
}
