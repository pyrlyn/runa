package runa.platform

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.cstr
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.set
import kotlinx.cinterop.toKString
import okio.FileSystem
import okio.IOException
import okio.Path
import platform.posix.F_TEST
import platform.posix.F_TLOCK
import platform.posix.O_APPEND
import platform.posix.O_CREAT
import platform.posix.O_RDWR
import platform.posix.O_WRONLY
import platform.posix.STDERR_FILENO
import platform.posix.STDIN_FILENO
import platform.posix.STDOUT_FILENO
import platform.posix._exit
import platform.posix.dup2
import platform.posix.errno
import platform.posix.execv
import platform.posix.fork
import platform.posix.getpid
import platform.posix.lockf
import platform.posix.open
import platform.posix.setsid
import platform.posix.strerror
import platform.posix.waitpid

public actual fun currentProcessId(): Int = getpid()

@OptIn(ExperimentalForeignApi::class)
public actual fun spawnDetached(
    executable: Path,
    arguments: List<String>,
    output: Path,
) {
    output.parent?.let(FileSystem.SYSTEM::createDirectories)
    memScoped {
        // Everything the child needs is prepared here: between fork and exec only async-signal-safe
        // calls are allowed, and allocating is not one of them.
        val program = executable.toString().cstr.ptr
        val argv = allocArray<CPointerVar<ByteVar>>(arguments.size + 2)
        argv[0] = program
        arguments.forEachIndexed { index, argument -> argv[index + 1] = argument.cstr.ptr }
        argv[arguments.size + 1] = null
        val log = output.toString().cstr.ptr
        val nowhere = "/dev/null".cstr.ptr

        val child = fork()
        if (child < 0) throw IOException("cannot start $executable: ${strerror(errno)?.toKString()}")
        if (child == 0) {
            // A new session has no terminal; the second fork makes init the parent, so nobody has
            // to wait for the process and it can never get a terminal back.
            setsid()
            if (fork() != 0) _exit(0)
            dup2(open(nowhere, O_RDWR), STDIN_FILENO)
            val out = open(log, O_WRONLY or O_CREAT or O_APPEND, OWNER_READ_WRITE)
            dup2(out, STDOUT_FILENO)
            dup2(out, STDERR_FILENO)
            // The caller's other files and sockets are not the new process's business.
            for (descriptor in STDERR_FILENO + 1 until INHERITED_DESCRIPTORS) platform.posix.close(descriptor)
            execv(program, argv)
            _exit(EXEC_FAILED)
        }
        val status = alloc<IntVar>()
        waitpid(child, status.ptr, 0)
    }
}

public actual fun restrictToOwner(path: Path) {
    val directory = FileSystem.SYSTEM.metadataOrNull(path)?.isDirectory == true
    changeMode(path.toString(), if (directory) OWNER_ALL else OWNER_READ_WRITE)
}

// mode_t is 16 bits wide on macOS and 32 on Linux, and code shared by both cannot convert to it.
internal expect fun changeMode(
    path: String,
    mode: Int,
)

@OptIn(ExperimentalForeignApi::class)
internal actual fun acquireFileLock(path: Path): Int? {
    path.parent?.let(FileSystem.SYSTEM::createDirectories)
    val descriptor = open(path.toString(), O_RDWR or O_CREAT, OWNER_READ_WRITE)
    if (descriptor < 0) throw IOException("cannot open $path: ${strerror(errno)?.toKString()}")
    if (lockf(descriptor, F_TLOCK, 0) != 0) {
        platform.posix.close(descriptor)
        return null
    }
    return descriptor
}

internal actual fun releaseFileLock(handle: Int) {
    // Closing the descriptor releases the lock.
    platform.posix.close(handle)
}

internal actual fun isFileLocked(path: Path): Boolean {
    // Read-write, because lockf only looks at a descriptor it could lock.
    val descriptor = open(path.toString(), O_RDWR)
    if (descriptor < 0) return false
    return try {
        lockf(descriptor, F_TEST, 0) != 0
    } finally {
        platform.posix.close(descriptor)
    }
}

private const val OWNER_READ_WRITE = 384 // 0600
private const val OWNER_ALL = 448 // 0700
private const val EXEC_FAILED = 127
private const val INHERITED_DESCRIPTORS = 256
