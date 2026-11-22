package runa.daemon

import runa.core.LogLevel
import kotlin.time.Clock

/**
 * The daemon's log: one line per message on stdout, which the process that started the daemon
 * pointed at `logs/daemon.log`.
 */
internal class DaemonLog(
    private val threshold: LogLevel,
    private val write: (String) -> Unit = ::println,
) {
    fun info(message: String) = at(LogLevel.INFO, message)

    fun error(message: String) = at(LogLevel.ERROR, message)

    fun at(
        level: LogLevel,
        message: String,
    ) {
        if (level >= threshold) write("${Clock.System.now()} ${level.id.uppercase().padEnd(LEVEL_WIDTH)} $message")
    }

    private companion object {
        const val LEVEL_WIDTH = 5
    }
}
