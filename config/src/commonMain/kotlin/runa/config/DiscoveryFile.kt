package runa.config

import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import okio.FileSystem
import okio.IOException
import okio.Path
import runa.platform.restrictToOwner
import runa.platform.writeAtomically

/** What a running daemon says about itself; [token] is the bearer token of its API. */
@Serializable
public data class DaemonInfo(
    val pid: Int,
    val port: Int,
    val token: String,
    val version: String,
)

/**
 * `run/daemon.json` (PLAN.md, section 7): written by the daemon when it listens, read by every
 * command that talks to it. The token in it is the only secret runa keeps, so the file and its
 * directory are the owner's alone.
 */
public class DiscoveryFile(
    private val fileSystem: FileSystem,
    public val path: Path,
) {
    /** Null when there is no file, or when it cannot be read: both mean no daemon can be found. */
    public fun read(): DaemonInfo? =
        try {
            if (fileSystem.exists(path)) format.decodeFromString(DaemonInfo.serializer(), fileSystem.read(path) { readUtf8() }) else null
        } catch (_: IOException) {
            null
        } catch (_: SerializationException) {
            null
        } catch (_: IllegalArgumentException) {
            null
        }

    /** @throws IOException when the file cannot be written. */
    public fun write(info: DaemonInfo) {
        val directory = checkNotNull(path.parent)
        fileSystem.createDirectories(directory)
        // Before the file exists, so the token is never readable by others, not even briefly.
        restrictToOwner(directory)
        fileSystem.writeAtomically(path) { writeUtf8(format.encodeToString(DaemonInfo.serializer(), info) + "\n") }
        restrictToOwner(path)
    }

    /** Removes the file if it still describes [pid]; a newer daemon's file is left alone. */
    public fun delete(pid: Int) {
        if (read()?.pid == pid) delete()
    }

    public fun delete() {
        try {
            fileSystem.delete(path, mustExist = false)
        } catch (_: IOException) {
            // Whoever reads it next finds a dead pid and tries again.
        }
    }

    private companion object {
        val format = Json { ignoreUnknownKeys = true }
    }
}
