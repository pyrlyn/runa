package runa.cli.output

import com.github.ajalt.mordant.terminal.Terminal
import runa.core.LogLevel

/** Progress notes and debug detail on stderr, so stdout stays clean for pipes and `--json`. */
internal class Diagnostics(
    private val terminal: Terminal,
    threshold: () -> LogLevel,
) {
    private val threshold by lazy(threshold)

    fun debug(message: () -> String) = log(LogLevel.DEBUG) { terminal.theme.muted(message()) }

    fun info(message: () -> String) = log(LogLevel.INFO, message)

    fun warn(message: () -> String) = log(LogLevel.WARN) { terminal.theme.warning("Warning:") + " " + message() }

    private fun log(
        level: LogLevel,
        message: () -> String,
    ) {
        if (level >= threshold) terminal.println(message(), stderr = true)
    }
}
