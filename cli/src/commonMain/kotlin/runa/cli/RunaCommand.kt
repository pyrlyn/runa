package runa.cli

import com.github.ajalt.clikt.completion.CompletionCommand
import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.CoreCliktCommand
import com.github.ajalt.clikt.core.UsageError
import com.github.ajalt.clikt.core.context
import com.github.ajalt.clikt.core.obj
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.versionOption
import com.github.ajalt.mordant.terminal.Terminal
import runa.cli.commands.ConfigCommand
import runa.cli.commands.ImportCommand
import runa.cli.commands.LsCommand
import runa.cli.commands.VersionCommand
import runa.config.ConfigKey

/** The root `runa` command: global options and the command tree. */
internal class RunaCommand(
    private val environment: CliEnvironment,
    terminal: Terminal,
    extraCommands: List<CoreCliktCommand> = emptyList(),
) : CliktCommand(name = "runa") {
    private val json by option("--json", help = "Print JSON instead of tables and text").flag()
    private val logLevel by option("--log-level", metavar = "level", help = "debug, info, warn or error (env: RUNA_LOG_LEVEL)")
    private val quiet by option("--quiet", help = "Print errors only").flag()
    private val verbose by option("--verbose", help = "Print debug detail").flag()
    private val host by option("--host", metavar = "host", help = "Daemon host (env: RUNA_HOST)")
    private val port by option("--port", metavar = "port", help = "Daemon port (env: RUNA_PORT)")
    private val backend by option("--backend", metavar = "name", help = "runa or lmstudio (env: RUNA_BACKEND)")
    private val yes by option("-y", "--yes", help = "Take the default answer for every prompt").flag()

    init {
        versionOption(environment.version, names = setOf("-v", "--version"), message = { "runa $it (${environment.gitSha})" })
        context {
            this.terminal = terminal
            helpFormatter = { GroupedHelpFormatter(it) }
        }
        subcommands(
            listOf(
                LsCommand(),
                ImportCommand(),
                VersionCommand(),
                ConfigCommand(),
                CompletionCommand(name = "completion", help = "Print a completion script for bash, zsh or fish"),
            ) + extraCommands,
        )
    }

    override val printHelpOnEmptyArgs: Boolean = true

    override fun help(context: Context): String = "Local models from the terminal."

    override fun run() {
        if (quiet && verbose) throw UsageError("--quiet and --verbose cannot be used together")
        val flags =
            listOfNotNull(
                logLevel?.let { ConfigKey.LOG_LEVEL to it },
                host?.let { ConfigKey.HOST to it },
                port?.let { ConfigKey.PORT to it },
                backend?.let { ConfigKey.BACKEND to it },
            ).toMap()
        val globals = Globals(environment, terminal, json, quiet, verbose, yes, flags)
        globals.config.checkOverrides()
        currentContext.obj = globals
    }
}
