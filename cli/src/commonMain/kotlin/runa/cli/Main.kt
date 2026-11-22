package runa.cli

import com.github.ajalt.clikt.core.CliktError
import com.github.ajalt.clikt.core.CoreCliktCommand
import com.github.ajalt.clikt.core.UsageError
import com.github.ajalt.clikt.core.parse
import com.github.ajalt.mordant.terminal.Terminal
import runa.core.BackendError
import runa.core.ExitCode
import runa.core.UserInputError

/** Runs `runa` with [args] and returns the process exit code (PLAN.md, phase 1, step 3). */
public fun runCli(
    args: List<String>,
    environment: CliEnvironment,
): Int = runCli(args, environment, extraCommands = emptyList())

internal fun runCli(
    args: List<String>,
    environment: CliEnvironment,
    extraCommands: List<CoreCliktCommand>,
): Int {
    // Piped output has no width to fit, so tables are never truncated there.
    val terminal = environment.terminal ?: Terminal(nonInteractiveWidth = PIPE_WIDTH)
    val command = RunaCommand(environment, terminal, extraCommands)

    fun fail(message: String?) = terminal.println(terminal.theme.danger("Error:") + " " + message, stderr = true)

    return try {
        command.parse(args)
        ExitCode.OK
    } catch (e: CliktError) {
        // Clikt uses this for --help and --version (status 0) as well as for usage mistakes.
        command.echoFormattedHelp(e)
        if (e is UsageError) ExitCode.USER_INPUT else e.statusCode
    } catch (e: UserInputError) {
        fail(e.message)
        ExitCode.USER_INPUT
    } catch (e: BackendError) {
        fail(e.message)
        ExitCode.BACKEND
    } catch (e: Exception) {
        // Anything else is a bug in runa: show where it came from.
        terminal.println(e.stackTraceToString(), stderr = true)
        ExitCode.UNEXPECTED
    }
}

private const val PIPE_WIDTH: Int = 10_000
