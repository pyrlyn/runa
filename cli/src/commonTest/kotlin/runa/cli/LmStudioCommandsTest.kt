package runa.cli

import okio.FileSystem
import runa.cli.FakeLmStudio.Companion.EMBEDDING
import runa.cli.FakeLmStudio.Companion.TINY
import runa.testing.assertGolden
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse

/**
 * The commands with `--backend lmstudio`, against [FakeLmStudio]. No test here has a daemon to
 * start, so a command that reached for runa's own would fail.
 */
class LmStudioCommandsTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val lmStudio = FakeLmStudio()

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    private fun run(
        vararg args: String,
        input: List<String> = emptyList(),
        interactive: Boolean = input.isNotEmpty(),
        env: Map<String, String> = emptyMap(),
        to: FakeLmStudio = lmStudio,
    ) = cli.run("--backend", "lmstudio", *args, input = input, interactive = interactive, env = env, http = to.http)

    @Test
    fun lsListsWhatLmStudioHas() {
        val run = run("ls")

        assertEquals("", run.stderr)
        assertGolden("lmstudio-ls.txt", run.stdout)
        assertGolden("lmstudio-ls.json", run("ls", "--json").stdout)
        assertEquals(listOf("GET /api/v1/models", "GET /api/v1/models"), lmStudio.requests)
    }

    @Test
    fun lsFiltersByKindAndName() {
        assertContains(run("ls", "--embedding").stdout, EMBEDDING)
        assertFalse(TINY in run("ls", "--embedding").stdout)
        assertFalse(EMBEDDING in run("ls", "tiny").stdout)

        val none = run("ls", "nope")
        assertEquals(2, none.exitCode)
        assertEquals("Error: No model matches 'nope' in LM Studio\n", none.stderr)
    }

    @Test
    fun lsHasNoVariantsToList() {
        val run = run("ls", "--variants")

        assertEquals(2, run.exitCode)
        assertEquals("Error: --variants is not supported by the lmstudio backend: LM Studio lists every model once\n", run.stderr)
        assertEquals(emptyList(), lmStudio.requests)
    }

    @Test
    fun psListsWhatIsLoadedWithoutTheColumnsLmStudioDoesNotTell() {
        val nothing = run("ps")
        assertEquals("", nothing.stdout)
        assertEquals("No models are loaded. Load one with: runa load <model>\n", nothing.stderr)

        lmStudio.loaded[TINY] = 4096
        lmStudio.loaded[EMBEDDING] = 512
        val run = run("ps")
        assertEquals("", run.stderr)
        assertGolden("lmstudio-ps.txt", run.stdout)
        assertGolden("lmstudio-ps.json", run("ps", "--json").stdout)
    }

    @Test
    fun loadPrintsTheInstanceAndNamesTheContextLength() {
        val run = run("load", "tiny", "--context-length", "1024")

        assertEquals(0, run.exitCode)
        assertEquals("$TINY\n", run.stdout)
        assertEquals("Loaded $TINY in 1.5 s: context 1024\n", run.stderr)
        assertEquals(
            listOf(
                // The whole key for a name without its publisher, then the instance that was loaded.
                "GET /api/v1/models",
                """POST /api/v1/models/load {"model":"$TINY","context_length":1024}""",
                "GET /api/v1/models",
            ),
            lmStudio.requests,
        )
    }

    @Test
    fun loadAsJsonSaysNullForTheEstimateLmStudioDoesNotMake() {
        val run = run("load", TINY, "--json")

        assertEquals(0, run.exitCode)
        assertGolden("lmstudio-load.json", run.stdout)
    }

    @Test
    fun loadRefusesTheSettingsLmStudioDoesNotTake() {
        for (args in listOf(
            listOf("--gpu", "max"),
            listOf("--ttl", "60"),
            listOf("--parallel", "2"),
            listOf("--identifier", "x"),
            listOf("--estimate-only"),
        )) {
            val run = run("load", TINY, *args.toTypedArray())

            assertEquals(2, run.exitCode, "args: $args")
            assertContains(run.stderr, "LM Studio's REST API loads a model with its context length and nothing else: it cannot take ")
        }
        assertEquals(emptyList(), lmStudio.requests)
    }

    @Test
    fun loadOfAModelLmStudioDoesNotHaveIsItsToExplain() {
        val run = run("load", "no/such-model")

        assertEquals(2, run.exitCode)
        assertEquals("Error: Model no/such-model not found in downloaded models\n", run.stderr)
    }

    @Test
    fun unloadFreesTheOneLoadedModel() {
        lmStudio.loaded[TINY] = 2048
        val run = run("unload")

        assertEquals("$TINY\n", run.stdout)
        assertEquals("Unloaded $TINY\n", run.stderr)
        assertEquals("""POST /api/v1/models/unload {"instance_id":"$TINY"}""", lmStudio.requests.last())
        assertEquals("No models are loaded\n", run("unload").stderr)
    }

    @Test
    fun unloadAllTakesOneRequestPerInstance() {
        lmStudio.loaded[TINY] = 2048
        lmStudio.loaded[EMBEDDING] = 512
        val run = run("unload", "--all", "--json")

        assertEquals("""{"unloaded":["$TINY","$EMBEDDING"]}""" + "\n", run.stdout)
        assertEquals(2, lmStudio.requests.count { it.startsWith("POST /api/v1/models/unload") })
        assertEquals(emptyMap(), lmStudio.loaded)
    }

    @Test
    fun unloadOfWhatIsNotLoadedIsLmStudiosToExplain() {
        val run = run("unload", "nothing-here")

        assertEquals(2, run.exitCode)
        assertEquals("Error: Model with instance identifier 'nothing-here' is not loaded.\n", run.stderr)
    }

    @Test
    fun chatLoadsTheModelAndPrintsTheAnswerWithoutItsLeadingEmptyLines() {
        val run = run("chat", "tiny", "-p", "hi")

        assertEquals(0, run.exitCode)
        assertEquals("echo: hi\n", run.stdout)
        assertEquals("Loaded $TINY in 1.5 s: context 2048\n", run.stderr)
        // Not stored: nothing follows the one answer.
        assertEquals(
            """POST /api/v1/chat {"model":"$TINY","input":"hi","system_prompt":"","stream":true,"store":false}""",
            lmStudio.requests.last(),
        )
    }

    @Test
    fun chatAsJsonIsTheWholeAnswerWithItsReasoning() {
        lmStudio.loaded[TINY] = 2048
        val run = run("chat", "-p", "hi", "-s", "Be brief", "--json")

        assertEquals("", run.stderr)
        assertGolden("lmstudio-chat.json", run.stdout)
    }

    @Test
    fun chatStatsLeaveOutTheStopReasonLmStudioDoesNotTell() {
        lmStudio.loaded[TINY] = 2048
        val run = run("chat", "-p", "hi", "--stats")

        assertEquals("prompt: 9 tokens, answer: 12 tokens, time to first token: 0.21 s, 29.75 tokens/s\n", run.stderr)
    }

    @Test
    fun aConversationGoesOnByTheIdOfTheLastAnswer() {
        lmStudio.loaded[TINY] = 2048
        val run = run("chat", input = listOf("hi", "and then?", "/exit"))

        assertEquals(0, run.exitCode)
        assertEquals("echo: hi\necho: and then? (after resp_1)\n", run.stdout)
        assertEquals(
            listOf(
                """POST /api/v1/chat {"model":"$TINY","input":"hi","system_prompt":"","stream":true}""",
                """POST /api/v1/chat {"model":"$TINY","input":"and then?","system_prompt":"","stream":true,"previous_response_id":"resp_1"}""",
            ),
            lmStudio.requests.filter { it.startsWith("POST") },
        )
    }

    @Test
    fun getHasLmStudioDownloadAndWaitsForIt() {
        val run = run("get", "ibm/granite-4-micro")

        assertEquals(0, run.exitCode)
        assertEquals("completed\n", run.stdout)
        assertEquals("LM Studio downloads ibm/granite-4-micro (1.00 KB)\nLM Studio downloaded ibm/granite-4-micro\n", run.stderr)
        assertEquals(
            listOf(
                """POST /api/v1/models/download {"model":"ibm/granite-4-micro"}""",
                "GET /api/v1/models/download/status/job_1",
                "GET /api/v1/models/download/status/job_1",
            ),
            lmStudio.requests,
        )
    }

    @Test
    fun getNamesAQuantizationBesideAHuggingFaceLink() {
        run("get", "acme/tiny-GGUF@Q4_K_M")
        run("get", "https://huggingface.co/acme/tiny-GGUF")

        val downloads = lmStudio.requests.filter { it.startsWith("POST") }
        assertEquals(
            listOf(
                """POST /api/v1/models/download {"model":"https://huggingface.co/acme/tiny-GGUF","quantization":"Q4_K_M"}""",
                """POST /api/v1/models/download {"model":"https://huggingface.co/acme/tiny-GGUF"}""",
            ),
            downloads,
        )
    }

    @Test
    fun getOfWhatLmStudioHasAsksNothingMore() {
        val run = run("get", TINY, "--json")

        assertEquals(0, run.exitCode)
        assertGolden("lmstudio-get-already.json", run.stdout)
        assertEquals("$TINY is already in LM Studio\n", run.stderr)
        assertEquals(1, lmStudio.requests.size)
    }

    @Test
    fun aDownloadThatFailsIsAFailure() {
        val run = run("get", "ibm/granite-4-micro", to = FakeLmStudio(downloadEnds = "failed"))

        assertEquals(3, run.exitCode)
        assertContains(run.stderr, "Error: LM Studio could not download ibm/granite-4-micro: the download is failed\n")
        assertEquals("", run.stdout)
    }

    @Test
    fun aPausedDownloadIsLeftToLmStudio() {
        val run = run("get", "ibm/granite-4-micro", to = FakeLmStudio(downloadEnds = "paused"))

        assertEquals(0, run.exitCode)
        assertEquals("paused\n", run.stdout)
        assertContains(run.stderr, "is paused in LM Studio")
    }

    @Test
    fun getCannotAskWhichVariant() {
        val run = run("get", "acme/tiny-GGUF", "--select")

        assertEquals(2, run.exitCode)
        assertEquals("Error: --select is not supported by the lmstudio backend: name the variant as owner/repo@quantization\n", run.stderr)
        assertEquals(emptyList(), lmStudio.requests)
    }

    @Test
    fun whatOnlyRunaDoesSaysSoAndAsksLmStudioNothing() {
        val commands =
            listOf(
                listOf("run", "x", "hi"),
                listOf("import", "x.gguf"),
                listOf("status"),
                listOf("daemon", "up"),
                listOf("daemon", "down"),
                listOf("daemon", "status"),
                listOf("server", "start"),
                listOf("server", "stop"),
                listOf("server", "status"),
                listOf("log", "stream"),
            )
        for (command in commands) {
            val run = run(*command.toTypedArray())
            val name = command.takeWhile { it != "x" && it != "x.gguf" }.joinToString(" ")

            assertEquals(2, run.exitCode, "command: $name")
            assertEquals(
                "Error: `runa $name` is not supported by the lmstudio backend; `--backend runa` runs it with runa itself\n",
                run.stderr,
            )
            assertEquals("", run.stdout)
        }
        assertEquals(emptyList(), lmStudio.requests)
        assertFalse(FileSystem.SYSTEM.exists(cli.home / "run"))
    }

    @Test
    fun whatNeedsNoBackendStillRuns() {
        assertEquals(0, run("version").exitCode)
        assertEquals("lmstudio\n", run("config", "get", "backend").stdout)
        assertEquals(emptyList(), lmStudio.requests)
    }

    @Test
    fun theBackendAlsoComesFromTheEnvironment() {
        val run = cli.run("ls", env = mapOf("RUNA_BACKEND" to "lmstudio"), http = lmStudio.http)

        assertEquals(0, run.exitCode)
        assertContains(run.stdout, TINY)
    }

    @Test
    fun anUnknownBackendIsAMistake() {
        val run = cli.run("--backend", "ollama", "ls", http = lmStudio.http)

        assertEquals(2, run.exitCode)
        assertEquals("Error: --backend must be one of: runa, lmstudio\n", run.stderr)
        assertEquals(emptyList(), lmStudio.requests)
    }

    @Test
    fun sendsLmStudiosTokenAndNeverTheDaemons() {
        run("--token", "from-flag", "ps")
        run("ps", env = mapOf("LM_API_TOKEN" to "from-env"))
        run("ps", env = mapOf("RUNA_TOKEN" to "the-daemons"))

        assertEquals(listOf("Bearer from-flag", "Bearer from-env", null), lmStudio.tokens)
    }

    @Test
    fun talksToTheHostAndPortItIsGiven() {
        val elsewhere = FakeLmStudio(port = 4321)

        assertEquals(0, run("--port", "4321", "ps", to = elsewhere).exitCode)
        assertEquals(0, run("ps", env = mapOf("RUNA_PORT" to "4321"), to = elsewhere).exitCode)

        val nobody = run("ps", to = elsewhere)
        assertEquals(3, nobody.exitCode)
        assertEquals("Error: Cannot reach LM Studio at http://127.0.0.1:1234: Connection refused\n", nobody.stderr)
    }
}
