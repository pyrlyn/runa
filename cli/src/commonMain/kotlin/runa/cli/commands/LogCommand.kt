package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.UsageError
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.choice
import kotlinx.coroutines.runBlocking
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.DaemonConnector
import runa.core.api.DaemonJson
import runa.core.api.LogEvent

/** One event as `log stream` prints it for a reader; [withStats] adds how a generation went. */
internal fun LogEvent.describe(withStats: Boolean): String {
    if (source != LogEvent.MODEL) return "$time ${level.orEmpty().uppercase().padEnd(LEVEL_WIDTH)} $text"
    val calls = toolCalls.orEmpty().map { "${it.name}(${it.arguments})" }
    // A prompt ends with a line break more often than not; --json has the text to the letter.
    val body = listOf(text.trimEnd('\n', '\r')).filter { it.isNotEmpty() } + calls + listOfNotNull(stats?.takeIf { withStats }?.summary())
    // A blank line after it: a prompt has many lines, and the next event must be easy to find.
    return (listOf("$time $endpoint '$model' $type") + body).joinToString("\n", postfix = "\n")
}

private const val LEVEL_WIDTH = 5

internal class LogCommand : CliktCommand(name = "log") {
    init {
        subcommands(Stream())
    }

    override val helpTags: Map<String, String> = CommandGroup.SERVE.helpTags
    override val printHelpOnEmptyArgs: Boolean = true

    override fun help(context: Context): String = "Watch what the daemon and its models do"

    override fun run() = Unit

    private class Stream : CliktCommand(name = "stream") {
        private val globals by requireObject<Globals>()
        private val source by option(
            "--source",
            help = "model: what the models read and write; server: the daemon's log and its requests; runtime: the engine's log",
        ).choice(*LogEvent.SOURCES.toTypedArray()).default(LogEvent.MODEL)
        private val filter by option("--filter", help = "Of the model source, only what the models read, or only what they write")
            .choice(*LogEvent.FILTERS.toTypedArray())
        private val stats by option("--stats", help = "Show token counts, time to first token and tokens per second of every answer").flag()
        private val json by option("--json", help = "Print every event as one line of JSON").flag()

        override fun help(context: Context): String =
            """
            Print the daemon's log as it is written, until Ctrl-C or until the daemon stops

            Nothing of what the models read and write is kept: the stream shows what happens while it is
            open. It reaches the daemon's log file only when the daemon was started with `--log-prompts`.
            """.trimIndent()

        override fun run() {
            if (filter != null && source != LogEvent.MODEL) throw UsageError("--filter is for --source ${LogEvent.MODEL}")
            val asJson = globals.wantsJson(json)
            runBlocking {
                DaemonConnector(globals).require().use { daemon ->
                    daemon.client.logs(
                        source = source,
                        filter = filter,
                        onListening = { globals.log.info { "Streaming the $source log of the daemon on ${daemon.address}; Ctrl-C stops" } },
                    ) { event ->
                        val line =
                            if (asJson) {
                                DaemonJson.encodeToString(LogEvent.serializer(), if (stats) event else event.copy(stats = null))
                            } else {
                                event.describe(stats)
                            }
                        // Not rendered: whatever a model wrote is shown as it wrote it.
                        terminal.rawPrint(line + "\n")
                    }
                }
            }
            globals.log.info { "The daemon ended the stream" }
        }
    }
}
