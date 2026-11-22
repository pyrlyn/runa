package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.UShortVar
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import platform.windows.GetCurrentProcessId
import platform.windows.GetModuleFileNameW
import platform.windows.MAX_PATH

public actual fun currentProcessId(): Int = GetCurrentProcessId().toInt()

@OptIn(ExperimentalForeignApi::class)
public actual fun executablePath(): Path =
    memScoped {
        val buffer = allocArray<UShortVar>(MAX_PATH)
        GetModuleFileNameW(null, buffer, MAX_PATH.toUInt())
        FileSystem.SYSTEM.canonicalize(buffer.toKString().toPath())
    }

// The daemon does not run on Windows yet (PLAN.md, phase 5): there is no llama.cpp build for it,
// so nothing could verify a CreateProcess and LockFileEx implementation.
public actual fun spawnDetached(
    executable: Path,
    arguments: List<String>,
    output: Path,
): Unit = throw UnsupportedOperationException(NO_DAEMON)

public actual fun restrictToOwner(path: Path) {
}

internal actual fun acquireFileLock(path: Path): Int? = throw UnsupportedOperationException(NO_DAEMON)

internal actual fun releaseFileLock(handle: Int) {
}

internal actual fun isFileLocked(path: Path): Boolean = false

private const val NO_DAEMON = "The runa daemon does not run on Windows yet"
