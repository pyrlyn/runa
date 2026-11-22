package runa.cli

import runa.testing.assertGolden
import kotlin.test.Test
import kotlin.test.assertEquals

class GoldenOutputTest {
    private val cli = TestCli()

    private fun golden(
        name: String,
        vararg args: String,
    ) {
        val run = cli.run(*args)
        assertEquals("", run.stderr)
        assertEquals(0, run.exitCode)
        assertGolden(name, run.stdout)
    }

    @Test
    fun rootHelp() = golden("help.txt", "--help")

    @Test
    fun configHelp() = golden("config-help.txt", "config", "--help")

    @Test
    fun configGetHelp() = golden("config-get-help.txt", "config", "get", "--help")

    @Test
    fun getHelp() = golden("get-help.txt", "get", "--help")

    @Test
    fun lsHelp() = golden("ls-help.txt", "ls", "--help")

    @Test
    fun importHelp() = golden("import-help.txt", "import", "--help")

    @Test
    fun runHelp() = golden("run-help.txt", "run", "--help")

    @Test
    fun chatHelp() = golden("chat-help.txt", "chat", "--help")

    @Test
    fun loadHelp() = golden("load-help.txt", "load", "--help")

    @Test
    fun unloadHelp() = golden("unload-help.txt", "unload", "--help")

    @Test
    fun psHelp() = golden("ps-help.txt", "ps", "--help")

    @Test
    fun statusHelp() = golden("status-help.txt", "status", "--help")

    @Test
    fun daemonHelp() = golden("daemon-help.txt", "daemon", "--help")

    @Test
    fun serverHelp() = golden("server-help.txt", "server", "--help")

    @Test
    fun serverStartHelp() = golden("server-start-help.txt", "server", "start", "--help")

    @Test
    fun versionJson() = golden("version.json", "version", "--json")

    @Test
    fun configGet() = golden("config-get.txt", "config", "get")

    @Test
    fun configGetJson() = golden("config-get.json", "config", "get", "--json")

    @Test
    fun configPathJson() = golden("config-path.json", "config", "path", "--json")
}
