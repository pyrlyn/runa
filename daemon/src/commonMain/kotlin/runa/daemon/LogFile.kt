package runa.daemon

import okio.FileSystem
import okio.IOException
import okio.Path
import okio.utf8Size
import runa.platform.appendStandardOutputTo
import runa.platform.standardOutputIs
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicLong
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/**
 * The daemon's log file. It is the daemon's standard output: whoever started the daemon pointed
 * that at [path], so what llama.cpp or a crash prints lands in the same file as the log.
 *
 * A file that has grown past [maxBytes] becomes `daemon.log.1`, the one before it `daemon.log.2`,
 * and so on for [kept] old files; the output then goes to a new file. The size is counted here and
 * not asked of the file system for every line, so what is printed past this class is not counted.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class LogFile(
    private val path: Path,
    private val fileSystem: FileSystem,
    private val maxBytes: Long = MAX_BYTES,
    private val kept: Int = KEPT_FILES,
    private val print: (String) -> Unit = ::println,
    private val reopen: (Path) -> Unit = ::appendStandardOutputTo,
) {
    private val written = AtomicLong(fileSystem.metadataOrNull(path)?.size ?: 0)
    private val rotating = AtomicBoolean(false)

    /** Told why the file could not be rotated. The log keeps going to the file it went to. */
    var onFailure: (String) -> Unit = {}

    /** Writes one line. May be called from any thread. */
    fun write(line: String) {
        print(line)
        if (written.addAndFetch(line.utf8Size() + 1) < maxBytes) return
        // A line that arrives from another thread meanwhile goes to the old file or the new one; either is fine.
        if (!rotating.compareAndSet(expectedValue = false, newValue = true)) return
        try {
            rotate()
            written.store(0)
        } catch (e: IOException) {
            // Counting starts again, so the next attempt comes after as much again and not with every line.
            written.store(0)
            onFailure("Cannot rotate $path: ${e.message}")
        } finally {
            rotating.store(false)
        }
    }

    private fun rotate() {
        fun old(number: Int) = checkNotNull(path.parent) / "${path.name}.$number"
        fileSystem.delete(old(kept), mustExist = false)
        for (number in kept - 1 downTo 1) {
            if (fileSystem.exists(old(number))) fileSystem.atomicMove(old(number), old(number + 1))
        }
        // The open file keeps taking the log under its new name until the output is pointed at a new one.
        fileSystem.atomicMove(path, old(1))
        reopen(path)
    }

    companion object {
        const val MAX_BYTES = 10L * 1024 * 1024
        const val KEPT_FILES = 3

        /** The log file at [path] when this process writes its output there, else null: a terminal is not rotated. */
        fun ofStandardOutput(
            path: Path,
            fileSystem: FileSystem,
        ): LogFile? = if (standardOutputIs(path)) LogFile(path, fileSystem) else null
    }
}
