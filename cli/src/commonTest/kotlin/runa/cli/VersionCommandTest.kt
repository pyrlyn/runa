package runa.cli

import kotlin.test.Test
import kotlin.test.assertEquals

class VersionCommandTest {
    private val cli = TestCli()

    @Test
    fun printsVersionAndCommit() {
        for (args in listOf(listOf("version"), listOf("--version"), listOf("-v"))) {
            val run = cli.run(*args.toTypedArray())
            assertEquals("runa 1.2.3 (abc1234)\n", run.stdout, "args: $args")
            assertEquals(0, run.exitCode)
        }
    }

    @Test
    fun jsonFlagWorksBeforeAndAfterTheCommand() {
        val expected = "{\"version\":\"1.2.3\",\"git_sha\":\"abc1234\"}\n"
        assertEquals(expected, cli.run("version", "--json").stdout)
        assertEquals(expected, cli.run("--json", "version").stdout)
    }

    @Test
    fun noArgumentsPrintsHelp() {
        val run = cli.run()
        assertEquals(cli.run("--help").stdout, run.stdout)
        assertEquals(0, run.exitCode)
    }
}
