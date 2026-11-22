package runa.cli

import com.github.ajalt.mordant.terminal.Terminal
import io.ktor.client.engine.HttpClientEngine
import okio.FileSystem
import runa.platform.HostOs
import runa.platform.hostOs
import runa.platform.physicalMemoryBytes
import runa.platform.systemEnv

/** Everything the CLI reads from the outside world, so tests can replace all of it. */
public class CliEnvironment internal constructor(
    internal val version: String,
    internal val gitSha: String,
    internal val env: (String) -> String?,
    internal val fileSystem: FileSystem,
    internal val os: HostOs,
    /** Null means: detect the real terminal (TTY, colours, `NO_COLOR`). */
    internal val terminal: Terminal?,
    /** Null means: the platform's own HTTP engine. */
    internal val httpEngine: HttpClientEngine?,
    /** The machine's memory in bytes, or null when the OS does not say. */
    internal val physicalMemory: () -> Long?,
) {
    /** The real process environment. Version and commit come from the executable's BuildInfo. */
    public constructor(version: String, gitSha: String) :
        this(version, gitSha, ::systemEnv, FileSystem.SYSTEM, hostOs, terminal = null, httpEngine = null, ::physicalMemoryBytes)
}
