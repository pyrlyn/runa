package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import okio.IOException
import okio.Path
import platform.windows.CreateHardLinkW
import platform.windows.GetLastError

@OptIn(ExperimentalForeignApi::class)
public actual fun createHardLink(
    target: Path,
    link: Path,
) {
    if (CreateHardLinkW(link.toString(), target.toString(), null) == 0) {
        throw IOException("cannot link $link to $target: Windows error ${GetLastError()}")
    }
}
