package runa.cli

import okio.FileSystem
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class ConfigCommandTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val configFile = cli.home / "config.json"

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    @Test
    fun pathIsUnderRunaHome() {
        assertEquals("$configFile\n", cli.run("config", "path").stdout)
    }

    @Test
    fun setWritesTheFileAndGetReadsIt() {
        val set = cli.run("config", "set", "server.port", "8080")
        assertEquals(0, set.exitCode)
        assertEquals("", set.stdout)
        assertEquals("Set server.port = 8080 in $configFile\n", set.stderr)

        assertEquals("8080\n", cli.run("config", "get", "server.port").stdout)
        assertEquals(
            "{\"key\":\"server.port\",\"value\":8080,\"source\":\"file\"}\n",
            cli.run("config", "get", "server.port", "--json").stdout,
        )
    }

    @Test
    fun precedenceIsFlagThenEnvThenFileThenDefault() {
        fun source(
            vararg globalFlags: String,
            env: Map<String, String> = emptyMap(),
        ) = cli.run(*globalFlags, "config", "get", "host", "--json", env = env).stdout.trim()

        assertEquals("""{"key":"host","value":"127.0.0.1","source":"default"}""", source())
        cli.run("config", "set", "host", "file.example")
        assertEquals("""{"key":"host","value":"file.example","source":"file"}""", source())
        val env = mapOf("RUNA_HOST" to "env.example")
        assertEquals("""{"key":"host","value":"env.example","source":"env"}""", source(env = env))
        assertEquals("""{"key":"host","value":"flag.example","source":"flag"}""", source("--host", "flag.example", env = env))
    }

    @Test
    fun quietHidesProgressAndVerboseAddsDetail() {
        assertEquals("", cli.run("--quiet", "config", "set", "ttl", "60").stderr)
        assertEquals("", cli.run("config", "get", "ttl").stderr)
        assertContains(cli.run("--verbose", "config", "get", "ttl").stderr, "Reading $configFile")
        assertContains(cli.run("config", "get", "ttl", env = mapOf("RUNA_LOG_LEVEL" to "debug")).stderr, "Reading")
        assertEquals("", cli.run("config", "set", "gpu", "off", env = mapOf("RUNA_LOG_LEVEL" to "warn")).stderr)
    }

    @Test
    fun invalidValueIsRejectedBeforeAnythingIsWritten() {
        val run = cli.run("config", "set", "server.port", "http")
        assertEquals(2, run.exitCode)
        assertEquals("Error: server.port must be a port number from 1 to 65535\n", run.stderr)
        assertFalse(FileSystem.SYSTEM.exists(configFile))

        assertEquals(2, cli.run("--log-level", "loud", "config", "set", "ttl", "5").exitCode)
        assertFalse(FileSystem.SYSTEM.exists(configFile))
    }

    @Test
    fun brokenFileFailsSettingsButNotVersionOrPath() {
        FileSystem.SYSTEM.createDirectories(cli.home)
        FileSystem.SYSTEM.write(configFile) { writeUtf8("{ not json") }

        val get = cli.run("config", "get")
        assertEquals(2, get.exitCode)
        assertContains(get.stderr, "Error: $configFile is not valid JSON")
        assertEquals(0, cli.run("version").exitCode)
        assertEquals(0, cli.run("config", "path").exitCode)
    }
}
