package runa.cli

import okio.FileSystem
import runa.config.DaemonSettings
import runa.core.api.DaemonJson
import runa.core.api.LogEvent
import runa.testing.assertGolden
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** `log stream` against [FakeDaemon], whose log has one event of every kind and then ends. */
class LogCommandTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val daemon = FakeDaemon(cli.home)

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    private fun run(vararg args: String) = cli.run(*args, daemon = daemon)

    private fun events(stdout: String) =
        stdout
            .lines()
            .filter {
                it.isNotEmpty()
            }.map { DaemonJson.decodeFromString(LogEvent.serializer(), it) }

    @Test
    fun streamShowsWhatTheModelsReadAndWrote() {
        val run = run("log", "stream")
        assertEquals(0, run.exitCode)
        assertEquals(
            "Starting the runa daemon\n" +
                "Streaming the model log of the daemon on 127.0.0.1:41234; Ctrl-C stops\n" +
                "The daemon ended the stream\n",
            run.stderr,
        )
        assertGolden("log-stream.txt", run.stdout)
        assertEquals(listOf("GET /api/v1/logs/stream?source=model"), daemon.requests)
    }

    @Test
    fun statsFollowAnAnswerWhenAskedFor() {
        val run = run("log", "stream", "--filter", "output", "--stats")
        assertEquals(0, run.exitCode)
        assertGolden("log-stream-stats.txt", run.stdout)
        assertEquals(listOf("GET /api/v1/logs/stream?source=model&filter=output"), daemon.requests)
    }

    @Test
    fun theOtherSourcesAreLinesOfALog() {
        assertEquals(
            "2026-01-02T03:04:05Z INFO  Loaded acme/alpha@Q4_K_M as 'acme/alpha@Q4_K_M' in 15 ms\n" +
                "2026-01-02T03:04:05Z DEBUG POST /api/v1/chat 200 310 ms\n",
            run("log", "stream", "--source", "server").stdout,
        )
        assertEquals("2026-01-02T03:04:05Z DEBUG llama_context: n_ctx = 4096\n", run("log", "stream", "--source", "runtime").stdout)
        assertEquals(listOf("GET /api/v1/logs/stream?source=server", "GET /api/v1/logs/stream?source=runtime"), daemon.requests)
    }

    @Test
    fun jsonIsOneEventALine() {
        val model = FakeDaemon.LOG.filter { it.source == LogEvent.MODEL }
        val plain = run("log", "stream", "--json")
        assertEquals(0, plain.exitCode)
        assertEquals(
            """{"time":"2026-01-02T03:04:05Z","source":"model","type":"input",""" +
                """"text":"<|user|>\nWhat is the weather?\n<|assistant|>\n","model":"acme/alpha@Q4_K_M","endpoint":"chat"}""",
            plain.stdout.lines().first(),
        )
        // How a generation went is left out until it is asked for, as it is without --json.
        assertEquals(model.map { it.copy(stats = null) }, events(plain.stdout))
        assertEquals(model, events(run("--json", "log", "stream", "--stats").stdout))
        assertEquals(
            FakeDaemon.LOG.filter { it.source == LogEvent.RUNTIME },
            events(run("log", "stream", "--source", "runtime", "--json").stdout),
        )
    }

    @Test
    fun aFilterBelongsToTheModelSource() {
        val run = run("log", "stream", "--source", "server", "--filter", "input")
        assertEquals(2, run.exitCode)
        assertContains(run.stderr, "--filter is for --source model")
        assertEquals(emptyList(), daemon.starts)

        val unknown = run("log", "stream", "--source", "prompts")
        assertEquals(2, unknown.exitCode)
        assertContains(unknown.stderr, "model, server, runtime")
    }

    @Test
    fun theDaemonIsToldToWriteTheConversationsDownWhenItIsStarted() {
        assertEquals(0, run("--log-prompts", "log", "stream").exitCode)
        assertEquals(listOf(listOf("--log-prompts", "__daemon")), daemon.starts)
    }

    @Test
    fun theDaemonProcessLearnsWhetherToWriteTheConversationsDown() {
        fun settings(
            vararg args: String,
            env: Map<String, String> = emptyMap(),
        ): DaemonSettings {
            var settings: DaemonSettings? = null
            val run =
                cli.run(*args, "__daemon", env = env, daemonProcess = {
                    settings = it
                    0
                })
            assertEquals(0, run.exitCode, run.stderr)
            return assertNotNull(settings)
        }
        assertFalse(settings().logPrompts)
        assertTrue(settings("--log-prompts").logPrompts)
        assertTrue(settings(env = mapOf("RUNA_LOG_PROMPTS" to "true")).logPrompts)
        assertEquals(0, cli.run("config", "set", "log_prompts", "true").exitCode)
        assertTrue(settings().logPrompts)
    }
}
