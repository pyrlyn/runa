package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.staticCFunction
import okio.Path
import platform.posix.SIGINT
import platform.posix.SIGTERM
import platform.posix.signal
import kotlin.concurrent.Volatile

public expect fun currentProcessId(): Int

/** The file this process was started from, with links resolved. */
public expect fun executablePath(): Path

/**
 * Starts [executable] with [arguments] as a process of its own: it has no terminal, belongs to no
 * shell job, and outlives the caller. Its output is appended to [output]. Returns once the process
 * is started, not when it is ready.
 *
 * @throws okio.IOException when the process cannot be started.
 */
public expect fun spawnDetached(
    executable: Path,
    arguments: List<String>,
    output: Path,
)

/** True when this process writes its standard output to the file at [path], and not to a terminal, a pipe or another file. */
public expect fun standardOutputIs(path: Path): Boolean

/**
 * Points this process's standard output at [path] from now on, appending to the file and creating
 * it for its owner alone. Standard error follows when it went where standard output went.
 *
 * @throws okio.IOException when the file cannot be opened.
 */
public expect fun appendStandardOutputTo(path: Path)

/** Makes [path] readable and writable by its owner only (a no-op on Windows, where ACLs inherit). */
public expect fun restrictToOwner(path: Path)

/**
 * An exclusive lock on a file, held until [close] or until the process ends, however it ends. That
 * is why it is a lock and not a pid file: a crashed owner never leaves it behind.
 */
public class FileLock private constructor(
    private val handle: Int,
) : AutoCloseable {
    override fun close(): Unit = releaseFileLock(handle)

    public companion object {
        /**
         * Locks [path], creating the file; null when another process holds the lock.
         *
         * @throws okio.IOException when the file cannot be opened.
         */
        public fun tryAcquire(path: Path): FileLock? = acquireFileLock(path)?.let(::FileLock)

        /**
         * True while another process holds the lock on [path]: the way to ask whether that process
         * is alive without trusting a process id, which the OS may have given to someone else.
         */
        public fun isHeld(path: Path): Boolean = isFileLocked(path)
    }
}

internal expect fun acquireFileLock(path: Path): Int?

internal expect fun releaseFileLock(handle: Int)

internal expect fun isFileLocked(path: Path): Boolean

/**
 * From now on SIGTERM and Ctrl-C only raise a flag, so a long-running process can stop in order.
 * A signal handler may do nothing more than that.
 */
@OptIn(ExperimentalForeignApi::class)
public fun catchTermination() {
    signal(SIGTERM, staticCFunction<Int, Unit> { terminationFlag = true })
    signal(SIGINT, staticCFunction<Int, Unit> { terminationFlag = true })
}

/** True once a signal caught by [catchTermination] has arrived. */
public fun terminationRequested(): Boolean = terminationFlag

@Volatile
private var terminationFlag: Boolean = false
