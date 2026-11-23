package runa.cli

import okio.FileSystem
import runa.cli.FakeDaemon.Companion.ALPHA
import runa.cli.FakeDaemon.Companion.BETA
import runa.core.ExitCode
import runa.testing.assertGolden
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** `runa chat` against [FakeDaemon], whose model answers with what it was asked. */
class ChatCommandTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val daemon = FakeDaemon(cli.home)

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    private fun run(
        vararg args: String,
        input: List<String> = emptyList(),
        interactive: Boolean = input.isNotEmpty(),
    ) = cli.run("chat", *args, daemon = daemon, input = input, interactive = interactive)

    private fun requests(path: String) = daemon.requests.filter { it.startsWith("POST $path ") }.map { it.substringAfter("$path ") }

    private fun chats() = requests("/api/v1/chat")

    private fun loads() = requests("/api/v1/models/load")

    @Test
    fun oneQuestionStartsTheDaemonLoadsTheModelAndPrintsTheAnswer() {
        val run = run("alpha", "-p", "hello there")
        assertEquals(0, run.exitCode, run.stderr)
        assertEquals("echo: hello there\n", run.stdout)
        assertEquals(
            "Starting the runa daemon\n" +
                "Loaded $ALPHA in 1.5 s: context 4096, gpu max, about 402 MB (weights 398 MB, context 4.10 MB)\n",
            run.stderr,
        )
        assertEquals(1, loads().size)
        // Nothing follows the one answer, so the daemon is not asked to keep it.
        assertContains(chats().single(), """"model":"$ALPHA"""")
        assertContains(chats().single(), """"store":false""")
        assertContains(chats().single(), """"previous_response_id":null""")
    }

    @Test
    fun theSystemPromptAndTheStatisticsOfOneQuestion() {
        val run = run("alpha", "-p", "hi", "-s", "be brief", "--stats", "--ttl", "60")
        assertEquals("[be brief] echo: hi\n", run.stdout)
        assertTrue(
            run.stderr.endsWith("prompt: 7 tokens, answer: 3 tokens (stop), time to first token: 0.25 s, 12.5 tokens/s\n"),
            run.stderr,
        )
        assertContains(loads().single(), """"ttl":60""")
    }

    @Test
    fun jsonHoldsTheWholeAnswer() {
        val run = cli.run("--quiet", "chat", "alpha", "-p", "hi", "--json", daemon = daemon)
        assertEquals("", run.stderr)
        assertGolden("chat.json", run.stdout)
    }

    @Test
    fun aQuestionCanBePipedIn() {
        val run = run("alpha", input = listOf("first line", "second line"), interactive = false)
        assertEquals("echo: first line\nsecond line\n", run.stdout)

        val nothing = run("alpha")
        assertEquals(ExitCode.USER_INPUT, nothing.exitCode)
        assertContains(nothing.stderr, "There is nothing to ask")
        assertEquals(1, chats().size)
    }

    @Test
    fun aLoadedModelIsNotLoadedAgain() {
        assertEquals(0, cli.run("load", "alpha", "--identifier", "mine", daemon = daemon).exitCode)
        // By its identifier, by the name of its model, and by being the only one.
        for (args in listOf(listOf("mine"), listOf("alpha"), listOf("acme/alpha"), emptyList())) {
            val run = run(*args.toTypedArray(), "-p", "hi")
            assertEquals("echo: hi\n", run.stdout, run.stderr)
            assertEquals("", run.stderr)
        }
        assertEquals(1, loads().size)
        assertTrue(chats().all { """"model":"mine"""" in it }, "${chats()}")
    }

    @Test
    fun withoutANameAndWithoutALoadedModelSomeoneHasToChoose() {
        val nobody = run("-p", "hi")
        assertEquals(ExitCode.USER_INPUT, nobody.exitCode)
        assertContains(nobody.stderr, "Name the model to load: $ALPHA, $BETA")

        // In a terminal the list of `load` is shown, and the chat goes on with the choice.
        val chosen = run(input = listOf("2", "hello", "/exit"))
        assertEquals(0, chosen.exitCode, chosen.stderr)
        assertContains(chosen.stdout, BETA)
        assertTrue(chosen.stdout.endsWith("echo: hello\n"), chosen.stdout)
        assertContains(chats().single(), """"model":"$BETA"""")
    }

    @Test
    fun aModelNobodyHasIsAMistake() {
        val run = run("gamma", "-p", "hi")
        assertEquals(ExitCode.USER_INPUT, run.exitCode)
        assertContains(run.stderr, "No model matches 'gamma'")
        assertEquals(emptyList(), chats())
    }

    @Test
    fun aConversationGoesOnUntilExit() {
        val run =
            run(
                "alpha",
                "--stats",
                input =
                    listOf(
                        "hello",
                        "more",
                        "/system-prompt be brief",
                        "and more",
                        "/clear",
                        "again",
                        "/stats off",
                        "\"\"\"two",
                        "lines\"\"\"",
                        "/stats",
                        "/nope",
                        "/exit",
                        "never read",
                    ),
            )
        assertEquals(0, run.exitCode, run.stderr)
        assertEquals(
            """
            echo: hello
            echo: more (after resp_1)
            [be brief] echo: and more (after resp_2)
            [be brief] echo: again
            [be brief] echo: two
            lines (after resp_4)

            """.trimIndent(),
            run.stdout,
        )
        assertGolden("chat-conversation.txt", run.stderr)
        assertEquals(5, chats().size)
        assertTrue(chats().all { """"store":true""" in it })
    }

    @Test
    fun theEndOfTheInputEndsTheChat() {
        val run = run("alpha", input = listOf("hello"))
        assertEquals(0, run.exitCode)
        assertEquals("echo: hello\n", run.stdout)
        // The shell's prompt starts on a line of its own.
        assertTrue(run.stderr.endsWith("> \n"), run.stderr)
    }

    @Test
    fun aFailureEndsOneAnswerAndNotTheChat() {
        val run = run("alpha", input = listOf("hello", FakeDaemon.FAILING, "more", "/exit"))
        assertEquals(0, run.exitCode, run.stderr)
        // What came before the failure stays, on a line of its own; the conversation continues where it was.
        assertEquals("echo: hello\nHalf an\necho: more (after resp_1)\n", run.stdout)
        assertContains(run.stderr, "Error: The answer was cut short: The model fell over")
    }

    @Test
    fun anotherModelTakesOverTheConversation() {
        val run = run("alpha", input = listOf("hello", "/model", "/model gamma", "/model beta", "more", "/exit"))
        assertEquals(0, run.exitCode, run.stderr)
        assertEquals("echo: hello\necho: more (after resp_1)\n", run.stdout)
        assertContains(run.stderr, "The model is $ALPHA\n")
        assertContains(run.stderr, "Error: No model matches 'gamma'")
        assertContains(run.stderr, "The model is $BETA\n")
        assertEquals(listOf(ALPHA, BETA), chats().map { it.substringAfter(""""model":"""").substringBefore('"') })
        assertEquals(2, daemon.instances.size)
    }

    @Test
    fun jsonNeedsOneQuestion() {
        val run = run("alpha", "--json", input = listOf("hello"))
        assertEquals(ExitCode.USER_INPUT, run.exitCode)
        assertContains(run.stderr, "--json needs one question")
        assertFalse(daemon.running)
    }
}
