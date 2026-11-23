package runa.cli

import okio.FileSystem
import runa.core.api.ServerDescription
import runa.testing.assertGolden
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** `server start`, `stop` and `status` against [FakeDaemon]. */
class ServerCommandTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val daemon = FakeDaemon(cli.home)

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    private fun run(
        vararg args: String,
        env: Map<String, String> = emptyMap(),
    ) = cli.run(*args, daemon = daemon, env = env)

    @Test
    fun startOpensTheServerOnce() {
        val first = run("server", "start")
        assertEquals(0, first.exitCode)
        assertEquals("Starting the runa daemon\n", first.stderr)
        assertEquals("The server is running on http://127.0.0.1:7862\n", first.stdout)
        assertEquals(
            """POST /api/v1/server {"running":true,"bind":"127.0.0.1","port":7862,"cors":false,"api_key":null}""",
            daemon.requests.single(),
        )

        val second = run("server", "start")
        assertEquals(0, second.exitCode)
        assertEquals("The server runs already\n", second.stderr)
        assertEquals(first.stdout, second.stdout)

        val moved = run("server", "start", "-p", "8000")
        assertEquals(2, moved.exitCode)
        assertContains(moved.stderr, "Error: The server already runs on 127.0.0.1:7862; stop it first with `runa server stop`")
        assertEquals(7862, daemon.server.port)
    }

    @Test
    fun flagsWinOverTheSettings() {
        assertEquals(0, cli.run("config", "set", "server.port", "9000").exitCode)
        assertEquals(0, cli.run("config", "set", "server.cors", "true").exitCode)
        assertEquals("The server is running on http://127.0.0.1:9000 (CORS)\n", run("server", "start").stdout)
        run("server", "stop")

        val flags = run("server", "start", "--port", "8000", "--bind", "::1", "--no-cors")
        assertEquals("The server is running on http://[::1]:8000\n", flags.stdout)
        assertEquals(ServerDescription(running = true, bind = "::1", port = 8000), daemon.server)
        run("server", "stop")

        assertTrue(run("server", "start", "--cors").stdout.endsWith("(CORS)\n"))
    }

    @Test
    fun aServerForOtherMachinesWithoutAKeyIsWarnedAbout() {
        val open = run("server", "start", "--bind", "0.0.0.0")
        assertEquals(0, open.exitCode)
        assertContains(open.stderr, "Warning: Other machines can reach 0.0.0.0:7862 and use the models without a key")
        assertEquals("The server is running on http://0.0.0.0:7862\n", open.stdout)
        run("server", "stop")

        val flag = run("server", "start", "--bind", "0.0.0.0", "--api-key", "s3cret")
        assertFalse("Warning" in flag.stderr, flag.stderr)
        assertEquals("The server is running on http://0.0.0.0:7862 (API key required)\n", flag.stdout)
        assertContains(daemon.requests.last(), """"api_key":"s3cret"""")
        // The key is the daemon's to keep: the CLI never prints it.
        assertFalse("s3cret" in flag.stdout + flag.stderr)
        run("server", "stop")

        val env = run("server", "start", "--bind", "0.0.0.0", env = mapOf("RUNA_API_KEY" to "from-env"))
        assertFalse("Warning" in env.stderr, env.stderr)
        assertContains(daemon.requests.last(), """"api_key":"from-env"""")
    }

    @Test
    fun whatCannotBeAServerAddressIsRefusedBeforeTheDaemonIsAsked() {
        for ((args, message) in listOf(
            listOf("-p", "0") to "--port must be a port number from 1 to 65535",
            listOf("--port", "http") to "--port must be a port number from 1 to 65535",
            listOf("--bind", "my host") to "--bind must be a host name or an IP address",
            listOf("--api-key", " ") to "--api-key must not be empty",
        )) {
            val run = run("server", "start", *args.toTypedArray())
            assertEquals(2, run.exitCode, "$args")
            assertContains(run.stderr, "Error: $message")
        }
        assertEquals(emptyList(), daemon.starts)
    }

    @Test
    fun stopClosesTheServerAndLeavesTheDaemon() {
        run("server", "start")
        val stop = run("server", "stop")
        assertEquals(0, stop.exitCode)
        assertEquals("Stopped the server on http://127.0.0.1:7862\n", stop.stderr)
        assertEquals("", stop.stdout)
        assertEquals(ServerDescription(), daemon.server)
        assertTrue(daemon.running)

        assertEquals("The server is not running\n", run("server", "stop").stderr)
    }

    @Test
    fun stopAndStatusNeverStartTheDaemon() {
        val stop = run("server", "stop")
        assertEquals(0, stop.exitCode)
        assertEquals("The server is not running\n", stop.stderr)

        val status = run("server", "status")
        assertEquals(0, status.exitCode)
        assertEquals("The server is not running\n", status.stdout)
        assertEquals("", status.stderr)
        assertGolden("server-status-stopped.json", run("server", "status", "--json").stdout)
        assertEquals(emptyList(), daemon.starts)
    }

    @Test
    fun statusSaysWhereTheServerIs() {
        run("server", "start", "--cors", "--api-key", "s3cret")
        val status = run("server", "status")
        assertEquals("The server is running on http://127.0.0.1:7862 (CORS, API key required)\n", status.stdout)
        assertEquals("", status.stderr)
        assertGolden("server-status.json", run("server", "status", "--json").stdout)
        assertGolden("server-start.json", run("server", "start", "--cors", "--api-key", "s3cret", "--json").stdout)
        assertContains(run("status").stdout, "Server: http://127.0.0.1:7862 (CORS, API key required)\n")
        assertContains(run("status", "--json").stdout, """"server":{"running":true,"bind":"127.0.0.1","port":7862,""")

        // A daemon that stops takes its server with it.
        run("daemon", "down")
        assertEquals("The server is not running\n", run("server", "status").stdout)
    }
}
