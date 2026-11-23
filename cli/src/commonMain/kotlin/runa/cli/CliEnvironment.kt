package runa.cli

import com.github.ajalt.mordant.terminal.Terminal
import io.ktor.client.engine.HttpClientEngine
import okio.FileSystem
import okio.Path
import runa.config.DaemonSettings
import runa.core.LogLevel
import runa.engine.Engine
import runa.platform.FileLock
import runa.platform.HardwareSurvey
import runa.platform.HostOs
import runa.platform.executablePath
import runa.platform.hostOs
import runa.platform.physicalMemoryBytes
import runa.platform.spawnDetached
import runa.platform.surveyHardware
import runa.platform.systemEnv
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

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
    /** What `runa runtime survey` shows of this machine. */
    internal val hardware: () -> HardwareSurvey,
    /** Creates the engine `runa run` loads models with; it gets the sink for the engine's own log. */
    internal val engine: (log: (LogLevel, String) -> Unit) -> Engine,
    /** Runs the daemon in this process until it is told to stop, and returns its exit code. */
    internal val daemon: (DaemonSettings) -> Int,
    /** Starts this executable with [arguments] as a background process whose output goes to [log]. */
    internal val startDaemon: (arguments: List<String>, log: Path) -> Unit = { arguments, log ->
        spawnDetached(executablePath(), arguments, log)
    },
    /** True while a daemon holds the lock file at the given path, that is, while one is alive. */
    internal val daemonLockHeld: (Path) -> Boolean = FileLock::isHeld,
    /** How long a command waits for a daemon it started (PLAN.md, section 5.1). */
    internal val daemonStartTimeout: Duration = 30.seconds,
    /** How often a download that LM Studio runs is asked how far it is. */
    internal val downloadPoll: Duration = 500.milliseconds,
) {
    /**
     * The real process environment. Version and commit come from the executable's BuildInfo, and
     * [engine] and [daemon] from the executable too: this module knows no inference backend and
     * does not contain the daemon.
     */
    public constructor(
        version: String,
        gitSha: String,
        engine: (log: (LogLevel, String) -> Unit) -> Engine,
        daemon: (DaemonSettings) -> Int,
    ) : this(
        version,
        gitSha,
        ::systemEnv,
        FileSystem.SYSTEM,
        hostOs,
        terminal = null,
        httpEngine = null,
        ::physicalMemoryBytes,
        ::surveyHardware,
        engine,
        daemon,
    )
}
