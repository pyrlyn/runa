package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.ProgramResult
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.mordant.widgets.Text
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.DaemonConnection
import runa.cli.daemon.DaemonConnector
import runa.cli.output.formatSeconds
import runa.core.BackendError
import runa.core.ExitCode
import runa.core.api.EngineDescription
import runa.core.api.ServerDescription
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

/** `runa status --json`: every field but [running] and [address] is null when no daemon runs. */
@Serializable
internal data class StatusView(
    val running: Boolean,
    /** Where the daemon is, or where it was looked for. */
    val address: String,
    val pid: Int? = null,
    val version: String? = null,
    @SerialName("uptime_seconds") val uptimeSeconds: Long? = null,
    val engine: EngineDescription? = null,
    @SerialName("loaded_instances") val loadedInstances: Int? = null,
    /** What `runa server start` opened. */
    val server: ServerDescription? = null,
) {
    constructor(daemon: DaemonConnection) : this(
        running = true,
        address = daemon.address,
        pid = daemon.status.pid,
        version = daemon.status.version,
        uptimeSeconds = daemon.status.uptimeSeconds,
        engine = daemon.status.engine,
        loadedInstances = daemon.status.loadedInstances,
        server = daemon.status.server,
    )

    fun describe(): String {
        if (!running || engine == null) return "The runa daemon is not running"
        return """
            The runa daemon is running on $address (pid $pid, runa $version, up ${formatSeconds(uptimeSeconds ?: 0)})
            Engine: ${engine.name} ${engine.version} (${engine.accelerators.joinToString(", ")})
            Loaded models: $loadedInstances
            Server: ${(server ?: ServerDescription()).describe()}
            """.trimIndent()
    }
}

/** Prints whether the daemon runs. It never starts one: asking must not change the answer. */
private fun printStatus(
    globals: Globals,
    json: Boolean,
) {
    val connector = DaemonConnector(globals)
    val view = runBlocking { connector.find()?.use(::StatusView) } ?: StatusView(running = false, address = connector.address)
    globals.renderer(json).render(view, StatusView.serializer()) { Text(it.describe()) }
}

internal class StatusCommand : CliktCommand(name = "status") {
    private val globals by requireObject<Globals>()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.SERVE.helpTags

    override fun help(context: Context): String = "Show whether the daemon runs and what it has loaded"

    override fun run() = printStatus(globals, json)
}

internal class DaemonCommand : CliktCommand(name = "daemon") {
    init {
        subcommands(Up(), Down(), Status())
    }

    override val helpTags: Map<String, String> = CommandGroup.SERVE.helpTags
    override val printHelpOnEmptyArgs: Boolean = true

    override fun help(context: Context): String =
        """
        Start and stop the background process that keeps models loaded

        `load`, `unload`, `ps` and `server` talk to the daemon, and `load`, `ps` and `server start` start it
        when it does not run.
        It listens on a loopback port of its own choosing; its log is logs/daemon.log in runa's directory.
        """.trimIndent()

    override fun run() = Unit

    private class Up : CliktCommand(name = "up") {
        private val globals by requireObject<Globals>()
        private val json by option("--json", help = JSON_HELP).flag()

        override fun help(context: Context): String = "Start the daemon, unless it runs already"

        override fun run() {
            val view =
                runBlocking {
                    DaemonConnector(globals).require().use { daemon ->
                        if (!daemon.started) globals.log.info { "The runa daemon runs already" }
                        StatusView(daemon)
                    }
                }
            globals.renderer(json).render(view, StatusView.serializer()) { Text(it.describe()) }
        }
    }

    private class Down : CliktCommand(name = "down") {
        private val globals by requireObject<Globals>()

        override fun help(context: Context): String = "Stop the daemon; the models it holds are unloaded"

        override fun run() =
            runBlocking {
                val connector = DaemonConnector(globals)
                val daemon = connector.find()
                if (daemon == null) {
                    globals.log.info { "The runa daemon is not running" }
                    return@runBlocking
                }
                val pid = daemon.use { it.client.shutdown().let { _ -> daemon.status.pid } }
                // Only this machine's daemon can be watched until it is really gone.
                if (!connector.byAddress) {
                    val asked = TimeSource.Monotonic.markNow()
                    while (connector.localDaemonAlive()) {
                        if (asked.elapsedNow() >
                            STOP_TIMEOUT
                        ) {
                            throw BackendError("The runa daemon (pid $pid) was asked to stop and still runs")
                        }
                        delay(POLL)
                    }
                }
                globals.log.info { "Stopped the runa daemon (pid $pid)" }
            }
    }

    private class Status : CliktCommand(name = "status") {
        private val globals by requireObject<Globals>()
        private val json by option("--json", help = JSON_HELP).flag()

        override fun help(context: Context): String = "Show whether the daemon runs; the same as `runa status`"

        override fun run() = printStatus(globals, json)
    }

    private companion object {
        val POLL = 50.milliseconds
        val STOP_TIMEOUT = 15.seconds
    }
}

/** `runa __daemon`: the daemon itself. `daemon up` starts this executable again with it. */
internal class DaemonProcessCommand : CliktCommand(name = DaemonConnector.DAEMON_COMMAND) {
    private val globals by requireObject<Globals>()

    override val hiddenFromHelp: Boolean = true

    override fun help(context: Context): String = "Run the daemon in the foreground; `runa daemon up` is the way to start it"

    override fun run() {
        val code = globals.environment.daemon(globals.daemonSettings())
        if (code != ExitCode.OK) throw ProgramResult(code)
    }
}
