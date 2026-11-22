package runa.cli.daemon

import kotlinx.coroutines.delay
import runa.cli.Globals
import runa.client.DaemonClient
import runa.config.ConfigKey
import runa.config.DiscoveryFile
import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.DaemonStatus
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

/** A daemon that answered its status probe; closing it closes the HTTP client. */
internal class DaemonConnection(
    val client: DaemonClient,
    val status: DaemonStatus,
    /** `host:port`, for messages. */
    val address: String,
    /** True when this command started the daemon. */
    val started: Boolean = false,
) : AutoCloseable {
    override fun close(): Unit = client.close()
}

/**
 * Finds the daemon a command should talk to, and starts one when the command needs it (PLAN.md,
 * section 5.1). This machine's daemon is found through its discovery file. A daemon given by
 * address, on another host or with `--port` and a token, is only ever connected to.
 */
internal class DaemonConnector(
    private val globals: Globals,
) {
    private val environment = globals.environment
    private val paths = globals.paths
    private val log = globals.log
    private val discovery = DiscoveryFile(environment.fileSystem, paths.discoveryFile)
    private val host = checkNotNull(globals.config.resolve(ConfigKey.HOST).value)
    private val port =
        globals.config
            .resolve(ConfigKey.PORT)
            .value
            ?.toInt()

    /** True when the daemon is named by address instead of being this machine's own. */
    val byAddress: Boolean = host.lowercase() !in LOOPBACK || (port != null && globals.token != null)

    /** Where the daemon is expected, for messages. */
    val address: String get() = if (port == null) host else "$host:$port"

    /** The running daemon, or null when there is none. Never starts one. */
    suspend fun find(): DaemonConnection? {
        if (byAddress) {
            val port = port ?: throw UserInputError("--host $host needs the daemon's port: --port or RUNA_PORT")
            val token =
                globals.token ?: throw UserInputError("A daemon on another machine needs its token: --token or ${Globals.TOKEN_ENV}")
            return probe(port, token, pid = null)
        }
        val info = discovery.read() ?: return null
        // The lock, not the pid: the OS may have given a dead daemon's pid to another process.
        if (!environment.daemonLockHeld(paths.daemonLockFile)) {
            log.debug { "Removing ${discovery.path}: the daemon that wrote it (pid ${info.pid}) is gone" }
            discovery.delete(info.pid)
            return null
        }
        if (port != null && port != info.port) {
            throw UserInputError(
                "The runa daemon runs on port ${info.port}, not $port: drop --port, or stop it first with `runa daemon down`",
            )
        }
        return probe(info.port, info.token, info.pid)
    }

    /** The running daemon; this machine's own is started first when it does not run. */
    suspend fun require(): DaemonConnection {
        find()?.let { return it }
        if (byAddress) throw BackendError("No runa daemon answers at $address")
        log.info { "Starting the runa daemon" }
        environment.fileSystem.createDirectories(checkNotNull(paths.daemonLogFile.parent))
        try {
            environment.startDaemon(globals.flagArguments(ConfigKey.PORT, ConfigKey.LOG_LEVEL) + DAEMON_COMMAND, paths.daemonLogFile)
        } catch (e: UnsupportedOperationException) {
            throw BackendError(e.message ?: "The runa daemon cannot run on this system", e)
        }
        val started = TimeSource.Monotonic.markNow()
        while (started.elapsedNow() < environment.daemonStartTimeout) {
            find()?.let { return DaemonConnection(it.client, it.status, it.address, started = true) }
            // A daemon takes the lock first thing, so no lock after a while means it has given up.
            if (started.elapsedNow() > LOCK_GRACE && !environment.daemonLockHeld(paths.daemonLockFile)) {
                throw BackendError("The runa daemon stopped while starting; its log is ${paths.daemonLogFile}")
            }
            delay(POLL)
        }
        throw BackendError("The runa daemon did not answer within ${environment.daemonStartTimeout}; its log is ${paths.daemonLogFile}")
    }

    /** True while this machine's daemon process is alive, whether or not it answers. */
    fun localDaemonAlive(): Boolean = environment.daemonLockHeld(paths.daemonLockFile)

    /** Null when nothing answers at [port], or when what answers is not the daemon with [pid]. */
    private suspend fun probe(
        port: Int,
        token: String,
        pid: Int?,
    ): DaemonConnection? {
        val address = "$host:$port"
        // An IPv6 address needs brackets in a URL.
        val client = DaemonClient(environment.httpEngine, "http://${if (':' in host) "[$host]" else host}:$port", token)
        val status =
            try {
                client.status(PROBE_TIMEOUT).takeIf { pid == null || it.pid == pid }
            } catch (e: BackendError) {
                log.debug { e.message.orEmpty() }
                null
            }
        if (status == null) {
            client.close()
            return null
        }
        if (status.version != environment.version) {
            log.warn {
                "The daemon at $address is runa ${status.version}, this is runa ${environment.version}; `runa daemon down` stops the old one"
            }
        }
        return DaemonConnection(client, status, address)
    }

    companion object {
        /** The hidden command that is the daemon process. */
        const val DAEMON_COMMAND: String = "__daemon"

        private val LOOPBACK = setOf("127.0.0.1", "localhost", "::1")
        private val POLL = 100.milliseconds
        private val PROBE_TIMEOUT = 2.seconds
        private val LOCK_GRACE = 5.seconds
    }
}
