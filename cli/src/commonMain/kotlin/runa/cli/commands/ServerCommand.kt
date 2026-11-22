package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.nullableFlag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.mordant.widgets.Text
import kotlinx.coroutines.runBlocking
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.DaemonConnector
import runa.config.ConfigKey
import runa.core.UserInputError
import runa.core.api.ServerDescription
import runa.core.api.ServerRequest
import runa.core.httpUrl
import runa.core.isLoopbackHost

/** `http://host:port`, or null when the server does not run. */
private fun ServerDescription.url(): String? {
    val bind = bind
    val port = port
    return if (running && bind != null && port != null) httpUrl(bind, port) else null
}

/** Where the server listens and what it asks of its callers, or that it does not run. */
internal fun ServerDescription.describe(): String {
    val url = url() ?: return "not running"
    val notes = listOfNotNull("CORS".takeIf { cors }, "API key required".takeIf { apiKeyRequired })
    return if (notes.isEmpty()) url else "$url (${notes.joinToString(", ")})"
}

internal class ServerCommand : CliktCommand(name = "server") {
    init {
        subcommands(Start(), Stop(), Status())
    }

    override val helpTags: Map<String, String> = CommandGroup.SERVE.helpTags
    override val printHelpOnEmptyArgs: Boolean = true

    override fun help(context: Context): String =
        """
        Serve the OpenAI-compatible API to other programs

        `server start` opens the daemon's API on an address and port of your choosing, by default
        127.0.0.1:7862. A program that speaks the OpenAI API takes `http://127.0.0.1:7862/v1` as its
        base URL; a model is loaded when a request first names it. `server stop` closes the port again
        and leaves the daemon and its models alone.
        """.trimIndent()

    override fun run() = Unit

    private class Start : CliktCommand(name = "start") {
        private val globals by requireObject<Globals>()
        private val port by option("-p", "--port", metavar = "port", help = "Port to listen on (config: server.port)")
        private val bind by option(
            "--bind",
            metavar = "address",
            help = "Address to listen on; 0.0.0.0 is every network interface (config: server.bind)",
        )
        private val cors by option("--cors", help = "Let pages in a browser call the API (config: server.cors)").nullableFlag("--no-cors")
        private val apiKey by option(
            "--api-key",
            metavar = "key",
            help = "Ask the callers of /v1 for this bearer key (env: $API_KEY_ENV)",
        )
        private val json by option("--json", help = JSON_HELP).flag()

        override fun help(context: Context): String = "Open the API on a port for other programs, unless it is open already"

        override fun run() {
            val key = apiKey ?: globals.environment.env(API_KEY_ENV)?.takeIf { it.isNotEmpty() }
            if (key != null && key.isBlank()) throw UserInputError("--api-key must not be empty")
            val request =
                ServerRequest(
                    running = true,
                    bind = setting(ConfigKey.SERVER_BIND, bind, "--bind"),
                    port = setting(ConfigKey.SERVER_PORT, port, "--port").toInt(),
                    cors = cors ?: (globals.config.resolve(ConfigKey.SERVER_CORS).value == "true"),
                    apiKey = key,
                )
            val server =
                runBlocking {
                    DaemonConnector(globals).require().use { daemon ->
                        // The daemon refuses to open a second one, so an answer here means it is the same.
                        val server = daemon.client.server(request)
                        if (daemon.status.server.running) globals.log.info { "The server runs already" }
                        server
                    }
                }
            if (!server.apiKeyRequired && !isLoopbackHost(request.bind.orEmpty())) {
                globals.log.warn {
                    "Other machines can reach ${request.bind}:${request.port} and use the models without a key; " +
                        "set one with --api-key or $API_KEY_ENV"
                }
            }
            globals.renderer(json).render(server, ServerDescription.serializer()) { Text("The server is running on ${it.describe()}") }
        }

        /** The flag when it was given, else the setting; a flag is checked as the setting would be. */
        private fun setting(
            key: ConfigKey,
            flag: String?,
            name: String,
        ): String {
            if (flag == null) return checkNotNull(globals.config.resolve(key).value)
            key.problem(flag)?.let { throw UserInputError("$name $it") }
            return flag
        }
    }

    private class Stop : CliktCommand(name = "stop") {
        private val globals by requireObject<Globals>()

        override fun help(context: Context): String = "Close the port; the daemon keeps running and keeps its models"

        override fun run() {
            // A daemon that does not run serves nothing, so there is no reason to start one.
            val stopped =
                runBlocking {
                    DaemonConnector(globals).find()?.use { daemon ->
                        daemon.status.server
                            .url()
                            ?.also { daemon.client.server(ServerRequest(running = false)) }
                    }
                }
            globals.log.info { if (stopped == null) "The server is not running" else "Stopped the server on $stopped" }
        }
    }

    private class Status : CliktCommand(name = "status") {
        private val globals by requireObject<Globals>()
        private val json by option("--json", help = JSON_HELP).flag()

        override fun help(context: Context): String = "Show whether the server runs and where"

        override fun run() {
            // Never starts the daemon: asking must not change the answer.
            val server = runBlocking { DaemonConnector(globals).find()?.use { it.status.server } } ?: ServerDescription()
            globals.renderer(json).render(server, ServerDescription.serializer()) {
                Text(if (it.running) "The server is running on ${it.describe()}" else "The server is not running")
            }
        }
    }

    private companion object {
        const val API_KEY_ENV = "RUNA_API_KEY"
    }
}
