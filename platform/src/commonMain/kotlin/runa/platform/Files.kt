package runa.platform

import okio.BufferedSink
import okio.FileSystem
import okio.Path

/**
 * Writes [path] through a temporary sibling and a rename, so a crash or a concurrent reader never
 * sees half a file. Creates the parent directories.
 *
 * @throws okio.IOException when the file cannot be written.
 */
public fun FileSystem.writeAtomically(
    path: Path,
    write: BufferedSink.() -> Unit,
) {
    val parent = checkNotNull(path.parent) { "$path has no parent directory" }
    val temporary = parent / "${path.name}.tmp"
    createDirectories(parent)
    write(temporary, writerAction = write)
    atomicMove(temporary, path)
}

/**
 * Makes [link] a second name for the file [target]; both must be on the same volume. Okio has no
 * call for this, so it goes to the operating system directly.
 *
 * @throws okio.IOException when the link cannot be created.
 */
public expect fun createHardLink(
    target: Path,
    link: Path,
)
