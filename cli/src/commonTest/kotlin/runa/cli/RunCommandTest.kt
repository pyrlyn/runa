package runa.cli

import okio.FileSystem
import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.engine.testing.FakeEngine
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** `runa run` against the fake engine, which answers with the prompt it was given. */
class RunCommandTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val models = cli.home / "models"
    private val engine = FakeEngine()
    private val file = writeModel(models / "acme" / "tiny" / "tiny-Q4_K_M.gguf")

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    private fun run(vararg args: String) = cli.run("run", *args, engine = engine)

    @Test
    fun streamsTheAnswerAndUnloadsTheModel() {
        val run = run("acme/tiny", "hello there")
        assertEquals(0, run.exitCode)
        assertEquals("user: hello there\n", run.stdout)
        assertEquals("", run.stderr)

        val load = engine.loads.single()
        assertEquals(file, load.path)
        assertEquals(LoadConfig(contextLength = 4096, gpu = GpuOffload.Max), load.config)
        assertTrue(load.model.closed)
    }

    @Test
    fun theSystemPromptComesFirst() {
        assertEquals("system: be brief user: hi\n", run("tiny", "hi", "--system", "be brief").stdout)
    }

    @Test
    fun loadOptionsComeFromFlagsThenTheConfigFile() {
        run("tiny", "hi", "--gpu", "0.5", "-c", "512")
        assertEquals(LoadConfig(contextLength = 512, gpu = GpuOffload.Ratio(0.5)), engine.loads.last().config)

        assertEquals(0, cli.run("config", "set", "gpu", "off").exitCode)
        assertEquals(0, cli.run("config", "set", "context_length", "2048").exitCode)
        run("tiny", "hi")
        assertEquals(LoadConfig(contextLength = 2048, gpu = GpuOffload.Off), engine.loads.last().config)
    }

    @Test
    fun jsonHoldsTheWholeAnswerAndItsStatistics() {
        val run = run("acme/tiny@Q4_K_M", "one two three", "--max-tokens", "3", "--json")
        assertEquals("", run.stderr)
        assertGolden("run.json", run.stdout)
    }

    @Test
    fun statsGoToStderr() {
        val run = run("tiny", "hi", "--stats")
        assertEquals("user: hi\n", run.stdout)
        assertEquals("prompt: 2 tokens, answer: 2 tokens (stop), time to first token: 0.01 s, 5.0 tokens/s\n", run.stderr)
    }

    @Test
    fun aModelFileCanBeNamedByItsPath() {
        val run = run(file.toString(), "hi")
        assertEquals("user: hi\n", run.stdout)
        assertEquals(file, engine.loads.single().path)
    }

    @Test
    fun anUnknownOrAmbiguousModelIsAUserError() {
        val unknown = run("nothing", "hi")
        assertEquals(2, unknown.exitCode)
        assertContains(unknown.stderr, "No model matches 'nothing'")

        writeModel(models / "acme" / "tiny" / "tiny-Q8_0.gguf", tinyLlama(fileType = 7))
        val ambiguous = run("tiny", "hi")
        assertEquals(2, ambiguous.exitCode)
        assertContains(ambiguous.stderr, "'tiny' matches 2 models, name one of them: acme/tiny@q4_k_m, acme/tiny@q8_0")
        assertTrue(engine.loads.isEmpty())
    }

    @Test
    fun badOptionsAreUsageErrors() {
        for (args in listOf(listOf("--gpu", "2"), listOf("--temp", "-1"), listOf("--max-tokens", "0"), listOf("-c", "0"))) {
            assertEquals(2, run("tiny", "hi", *args.toTypedArray()).exitCode, "$args")
        }
        assertEquals(2, run("tiny").exitCode)
    }
}
