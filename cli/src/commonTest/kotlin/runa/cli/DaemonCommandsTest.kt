package runa.cli

import okio.FileSystem
import runa.cli.FakeDaemon.Companion.ALPHA
import runa.cli.FakeDaemon.Companion.BETA
import runa.config.DaemonInfo
import runa.config.DaemonSettings
import runa.core.GpuOffload
import runa.core.LogLevel
import runa.testing.assertGolden
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

/** `daemon`, `status`, `ps`, `load` and `unload` against [FakeDaemon]. */
class DaemonCommandsTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val daemon = FakeDaemon(cli.home)

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    private fun run(
        vararg args: String,
        input: List<String> = emptyList(),
        env: Map<String, String> = emptyMap(),
    ) = cli.run(*args, daemon = daemon, input = input, env = env)

    @Test
    fun statusNeverStartsTheDaemon() {
        for (args in listOf(listOf("status"), listOf("daemon", "status"))) {
            val run = run(*args.toTypedArray())
            assertEquals(0, run.exitCode)
            assertEquals("The runa daemon is not running\n", run.stdout)
            assertEquals("", run.stderr)
        }
        assertGolden("status-stopped.json", run("status", "--json").stdout)
        assertEquals(emptyList(), daemon.starts)
        assertFalse(FileSystem.SYSTEM.exists(cli.home / "run"))
    }

    @Test
    fun statusDescribesARunningDaemon() {
        daemon.start()
        val run = run("status")
        assertEquals("", run.stderr)
        assertGolden("status.txt", run.stdout)
        assertGolden("status.json", run("daemon", "status", "--json").stdout)
        assertEquals(1, daemon.starts.size)
    }

    @Test
    fun daemonUpStartsItOnce() {
        val first = run("daemon", "up")
        assertEquals(0, first.exitCode)
        assertEquals("Starting the runa daemon\n", first.stderr)
        assertContains(first.stdout, "The runa daemon is running on 127.0.0.1:41234 (pid 4242, runa 1.2.3, up 2m)")

        val second = run("daemon", "up")
        assertEquals("The runa daemon runs already\n", second.stderr)
        assertEquals(listOf(listOf("__daemon")), daemon.starts)
    }

    @Test
    fun daemonDownStopsItAndIsFineWithoutOne() {
        daemon.start()
        val down = run("daemon", "down")
        assertEquals(0, down.exitCode)
        assertEquals("Stopped the runa daemon (pid 4242)\n", down.stderr)
        assertFalse(daemon.running)

        val again = run("daemon", "down")
        assertEquals(0, again.exitCode)
        assertEquals("The runa daemon is not running\n", again.stderr)
        assertEquals(1, daemon.starts.size)
    }

    @Test
    fun theFlagsThatMatterToTheDaemonArePassedOn() {
        val run = run("--port", "5000", "--log-level", "debug", "--quiet", "ps")
        assertEquals(0, run.exitCode)
        assertEquals(listOf(listOf("--port", "5000", "--log-level", "debug", "__daemon")), daemon.starts)
        assertEquals(5000, daemon.discovery.read()?.port)

        // The daemon runs where it was started; asking for another port does not move it.
        val moved = run("--port", "5001", "ps")
        assertEquals(2, moved.exitCode)
        assertContains(moved.stderr, "The runa daemon runs on port 5000, not 5001")
        assertEquals(1, daemon.starts.size)
    }

    @Test
    fun aStaleDiscoveryFileIsRemovedAndReplaced() {
        daemon.start()
        daemon.kill()
        assertNotNull(daemon.discovery.read())

        val status = run("status")
        assertEquals("The runa daemon is not running\n", status.stdout)
        assertNull(daemon.discovery.read())
        assertEquals(1, daemon.starts.size)

        // The same again, but with a command that needs the daemon: it starts a new one.
        daemon.discovery.write(DaemonInfo(pid = 99, port = 1, token = "old", version = "0.0.1"))
        val ps = run("ps")
        assertEquals(0, ps.exitCode)
        assertEquals("Starting the runa daemon\nNo models are loaded. Load one with: runa load <model>\n", ps.stderr)
        assertEquals(FakeDaemon.PID, daemon.discovery.read()?.pid)
        assertEquals(2, daemon.starts.size)
    }

    @Test
    fun aDaemonThatNeverAnswersIsABackendError() {
        val silent = FakeDaemon(cli.home, comesUp = false)
        val run = cli.run("ps", daemon = silent)
        assertEquals(3, run.exitCode)
        assertContains(run.stderr, "The runa daemon did not answer within 300ms; its log is ${cli.home / "logs" / "daemon.log"}")
        assertEquals(1, silent.starts.size)
    }

    @Test
    fun anotherVersionOfTheDaemonIsPointedOut() {
        val old = FakeDaemon(cli.home, version = "1.0.0")
        old.start()
        val run = cli.run("ps", daemon = old)
        assertEquals(0, run.exitCode)
        assertContains(run.stderr, "Warning: The daemon at 127.0.0.1:41234 is runa 1.0.0, this is runa 1.2.3")
    }

    @Test
    fun loadSendsItsOptionsAndPrintsTheIdentifier() {
        val run = run("load", "alpha", "--identifier", "mine", "-c", "512", "--gpu", "off", "--parallel", "2", "--ttl", "60")
        assertEquals(0, run.exitCode)
        assertEquals("mine\n", run.stdout)
        assertEquals(
            "Starting the runa daemon\n" +
                "Loaded $ALPHA in 1.5 s: context 512, gpu off, about 399 MB (weights 398 MB, context 512 KB)\n",
            run.stderr,
        )
        assertEquals(
            """POST /api/v1/models/load {"model":"alpha","context_length":512,"gpu":"off","parallel":2,"ttl":60,""" +
                """"identifier":"mine","estimate_only":false,"stream":true}""",
            daemon.requests.single(),
        )
    }

    @Test
    fun loadLeavesTheDefaultsToTheDaemon() {
        assertEquals("$ALPHA\n", run("load", "alpha").stdout)
        assertContains(
            daemon.requests.single(),
            """{"model":"alpha","context_length":null,"gpu":null,"parallel":null,"ttl":null,"identifier":null,""",
        )
        assertGolden("load.json", run("load", "beta", "--json").stdout)
    }

    @Test
    fun loadEstimatesWithoutLoading() {
        val run = run("load", "alpha", "--estimate-only", "-c", "2000")
        assertEquals("Estimated memory: 400 MB (weights 398 MB, context 2.00 MB)\n", run.stdout)
        assertGolden("load-estimate.json", run("load", "alpha", "--estimate-only", "--json").stdout)
        assertEquals(emptyList(), daemon.instances)
    }

    @Test
    fun loadWithoutAModelAsksWhenSomeoneCanAnswer() {
        val asked = run("load", input = listOf("7", "2"))
        assertEquals(0, asked.exitCode)
        assertContains(asked.stdout, "1  $ALPHA  llm   398 MB")
        assertContains(asked.stdout, "2  $BETA     llm   676 MB")
        assertTrue(asked.stdout.endsWith("$BETA\n"), asked.stdout)
        assertEquals(listOf(BETA), daemon.instances.map { it.model })

        // The list says what is loaded already, and Enter takes the first row.
        val again = run("load", input = listOf(""))
        assertContains(again.stdout, "676 MB  loaded")
        assertEquals(listOf(BETA, ALPHA), daemon.instances.map { it.model })
    }

    @Test
    fun loadWithoutAModelAndNobodyToAskNamesTheChoices() {
        for (args in listOf(listOf("load"), listOf("-y", "load"), listOf("load", "--json"))) {
            val run = run(*args.toTypedArray(), input = if ("-y" in args || "--json" in args) listOf("1") else emptyList())
            assertEquals(2, run.exitCode)
            assertContains(run.stderr, "Error: Name the model to load: $ALPHA, $BETA")
        }
        assertEquals(emptyList(), daemon.instances)
    }

    @Test
    fun aRefusedLoadIsTheUsersMistake() {
        val unknown = run("load", "gamma")
        assertEquals(2, unknown.exitCode)
        assertContains(unknown.stderr, "Error: No model matches 'gamma'")

        run("load", "alpha")
        val twice = run("load", "beta", "--identifier", ALPHA)
        assertEquals(2, twice.exitCode)
        assertContains(twice.stderr, "already loaded as '$ALPHA'")

        assertEquals(2, run("load", "alpha", "--gpu", "lots").exitCode)
        assertEquals(2, run("load", "alpha", "--ttl", "-1").exitCode)
        assertEquals(2, run("load", "alpha", "--parallel", "0").exitCode)
    }

    @Test
    fun psListsTheLoadedModels() {
        run("load", "alpha", "--ttl", "600")
        run("load", "beta", "--identifier", "big", "-c", "8192", "--gpu", "0.5", "--parallel", "4")
        val ps = run("ps")
        assertEquals("", ps.stderr)
        assertGolden("ps.txt", ps.stdout)
        assertGolden("ps.json", run("ps", "--json").stdout)
    }

    @Test
    fun psWithNothingLoadedIsAnEmptyList() {
        daemon.start()
        val text = run("ps")
        assertEquals("", text.stdout)
        assertEquals("No models are loaded. Load one with: runa load <model>\n", text.stderr)
        val json = run("ps", "--json")
        assertEquals("[]\n", json.stdout)
        assertEquals("", json.stderr)
    }

    @Test
    fun unloadTakesAnIdentifierOrTheOnlyModelOrAll() {
        run("load", "alpha")
        val only = run("unload")
        assertEquals("$ALPHA\n", only.stdout)
        assertEquals("Unloaded $ALPHA\n", only.stderr)

        run("load", "alpha")
        run("load", "beta", "--identifier", "big")
        val ambiguous = run("unload")
        assertEquals(2, ambiguous.exitCode)
        assertContains(ambiguous.stderr, "Error: Name the model to unload, or use --all: $ALPHA, big")

        assertEquals("big\n", run("unload", "big").stdout)
        assertEquals(2, run("unload", "big").exitCode)
        run("load", "beta")
        assertEquals("""{"unloaded":["$ALPHA","$BETA"]}""" + "\n", run("unload", "--all", "--json").stdout)
        assertEquals(emptyList(), daemon.instances)

        val nothing = run("unload")
        assertEquals(0, nothing.exitCode)
        assertEquals("No models are loaded\n", nothing.stderr)
        assertEquals("", nothing.stdout)
        assertEquals(2, run("unload", "big", "--all").exitCode)
    }

    @Test
    fun unloadAsksWhichModelWhenSomeoneCanAnswer() {
        run("load", "alpha")
        run("load", "beta", "--identifier", "big")
        val asked = run("unload", input = listOf("2"))
        assertContains(asked.stdout, "2  big")
        assertEquals(listOf(ALPHA), daemon.instances.map { it.id })
    }

    @Test
    fun unloadDoesNotStartADaemonToFindItEmpty() {
        val run = run("unload", "--all")
        assertEquals(0, run.exitCode)
        assertEquals("The runa daemon is not running, so no model is loaded\n", run.stderr)
        assertEquals("", run.stdout)
        assertEquals("""{"unloaded":[]}""" + "\n", run("unload", "--all", "--json").stdout)
        assertEquals(emptyList(), daemon.starts)
    }

    @Test
    fun aDaemonGivenByAddressIsNeverStarted() {
        val noPort = run("--host", "192.0.2.5", "ps")
        assertEquals(2, noPort.exitCode)
        assertContains(noPort.stderr, "--host 192.0.2.5 needs the daemon's port")

        val noToken = run("--host", "192.0.2.5", "--port", "41234", "ps")
        assertEquals(2, noToken.exitCode)
        assertContains(noToken.stderr, "needs its token: --token or RUNA_TOKEN")

        val down = run("--host", "192.0.2.5", "--port", "41234", "--token", FakeDaemon.TOKEN, "ps")
        assertEquals(3, down.exitCode)
        assertContains(down.stderr, "Error: No runa daemon answers at 192.0.2.5:41234")
        assertEquals("The runa daemon is not running\n", run("--host", "192.0.2.5", "--port", "41234", "--token", "x", "status").stdout)
        assertEquals(emptyList(), daemon.starts)
    }

    @Test
    fun aDaemonGivenByAddressIsReachedWithTheGivenToken() {
        daemon.start()
        // No discovery file on this machine: all the CLI knows is the address and the token.
        daemon.discovery.delete()
        val address = arrayOf("--host", "192.0.2.5", "--port", "41234")
        assertEquals("$ALPHA\n", run(*address, "load", "alpha", env = mapOf("RUNA_TOKEN" to FakeDaemon.TOKEN)).stdout)
        assertContains(run(*address, "--token", FakeDaemon.TOKEN, "ps").stdout, ALPHA)

        val wrong = run(*address, "--token", "wrong", "ps")
        assertEquals(3, wrong.exitCode)
        assertContains(wrong.stderr, "rejected the token")
        assertEquals(1, daemon.starts.size)
    }

    @Test
    fun theDaemonProcessGetsTheResolvedSettings() {
        assertEquals(0, cli.run("config", "set", "ttl", "90").exitCode)
        assertEquals(0, cli.run("config", "set", "gpu", "off").exitCode)
        var settings: DaemonSettings? = null
        val run =
            cli.run("--port", "5000", "--log-level", "warn", "__daemon", env = mapOf("RUNA_HOST" to "127.0.0.1"), daemonProcess = {
                settings = it
                0
            })
        assertEquals(0, run.exitCode)
        val got = assertNotNull(settings)
        assertEquals("1.2.3", got.version)
        assertEquals(cli.home / "models", got.modelsDir)
        assertEquals(cli.home / "run" / "daemon.json", got.paths.discoveryFile)
        assertEquals(5000, got.port)
        assertEquals(LogLevel.WARN, got.logLevel)
        assertEquals(null, got.contextLength)
        assertEquals(GpuOffload.Off, got.gpu)
        assertEquals(90.seconds, got.ttl)

        assertEquals(3, cli.run("__daemon", daemonProcess = { 3 }).exitCode)
        assertFalse("__daemon" in cli.run("--help").stdout)
    }
}
