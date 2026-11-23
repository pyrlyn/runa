package runa.cli

import com.github.ajalt.clikt.core.CliktCommand
import runa.core.BackendError
import runa.core.UserInputError
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class ExitCodeTest {
    private val cli = TestCli()

    private class Failing(
        private val error: Exception,
    ) : CliktCommand(name = "fail") {
        override fun run(): Unit = throw error
    }

    private fun fail(error: Exception) = cli.run("fail", extraCommands = listOf(Failing(error)))

    @Test
    fun successIsZero() {
        assertEquals(0, cli.run("version").exitCode)
        assertEquals(0, cli.run("--help").exitCode)
        assertEquals(0, cli.run("completion", "zsh").exitCode)
    }

    @Test
    fun usageMistakesExitWithTwoAndShowUsage() {
        val cases =
            listOf(
                listOf("--no-such-option"),
                listOf("no-such-command"),
                listOf("config", "get", "no.such.key"),
                listOf("config", "set", "port"),
                listOf("completion", "powershell"),
                listOf("--quiet", "--verbose", "version"),
            )
        for (args in cases) {
            val run = cli.run(*args.toTypedArray())
            assertEquals(2, run.exitCode, "args: $args")
            assertEquals("", run.stdout, "args: $args")
            assertContains(run.stderr, "Usage: runa")
            assertContains(run.stderr, "Error:")
        }
    }

    @Test
    fun userInputErrorIsOneLineAndTwo() {
        val run = fail(UserInputError("no model named qwen"))
        assertEquals(2, run.exitCode)
        assertEquals("Error: no model named qwen\n", run.stderr)
        assertEquals("", run.stdout)
    }

    @Test
    fun invalidOverridesAreUserInputErrors() {
        assertEquals("Error: --port must be a port number from 1 to 65535\n", cli.run("--port", "http", "version").stderr)
        val run = cli.run("version", env = mapOf("RUNA_BACKEND" to "ollama"))
        assertEquals(2, run.exitCode)
        assertEquals("Error: RUNA_BACKEND must be one of: runa, lmstudio\n", run.stderr)
    }

    @Test
    fun backendErrorIsOneLineAndThree() {
        val run = fail(BackendError("the daemon is not running"))
        assertEquals(3, run.exitCode)
        assertEquals("Error: the daemon is not running\n", run.stderr)
    }

    @Test
    fun unexpectedErrorPrintsAStackTraceAndOne() {
        val run = fail(IllegalStateException("boom"))
        assertEquals(1, run.exitCode)
        assertContains(run.stderr, "IllegalStateException: boom")
        assertContains(run.stderr, "    at ")
        assertFalse(run.stderr.startsWith("Error:"))
    }
}
