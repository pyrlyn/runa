package runa.cli

import runa.cli.commands.ChatAction
import runa.cli.commands.ChatState
import runa.cli.commands.PendingMessage
import runa.core.api.ChatResponse
import runa.core.api.ChatStats
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The rules of the input of `runa chat`, without a terminal or a daemon. */
class ChatInputTest {
    private val start = ChatState(model = "alpha")

    /** Types [lines] one after the other and returns the state and what each line led to. */
    private fun type(
        vararg lines: String,
        from: ChatState = start,
    ): Pair<ChatState, List<ChatAction>> {
        var state = from
        val actions =
            lines.map { line ->
                val step = state.read(line)
                state = step.state
                step.action
            }
        return state to actions
    }

    private fun sent(vararg lines: String): List<String> = type(*lines).second.filterIsInstance<ChatAction.Send>().map { it.text }

    private fun said(
        line: String,
        from: ChatState = start,
    ): String = assertIs<ChatAction.Say>(from.read(line).action).text

    @Test
    fun aLineIsAMessage() {
        val (state, actions) = type("  Hello there  ")
        assertEquals(listOf<ChatAction>(ChatAction.Send("Hello there")), actions)
        assertEquals(start, state)
    }

    @Test
    fun anEmptyLineIsNothing() {
        assertEquals(listOf<ChatAction>(ChatAction.Wait, ChatAction.Wait), type("", "   ").second)
    }

    @Test
    fun aBlockIsOneMessageOfSeveralLines() {
        val (state, actions) = type("\"\"\"", "first", "", "  second", "\"\"\"")
        assertEquals(List(4) { ChatAction.Wait } + ChatAction.Send("first\n\n  second"), actions)
        assertNull(state.pending)

        assertEquals(listOf("one\ntwo"), sent("\"\"\"one", "two\"\"\""))
        assertEquals(listOf("all in one line"), sent("\"\"\"all in one line\"\"\""))
        // Inside a block nothing is a command, which is also how a message can start with a slash.
        assertEquals(listOf("/exit\n/help"), sent("\"\"\"", "/exit", "/help", "\"\"\""))
    }

    @Test
    fun anOpenBlockIsKnownByItsState() {
        val (state, actions) = type("\"\"\"first", "second")
        assertEquals(listOf<ChatAction>(ChatAction.Wait, ChatAction.Wait), actions)
        assertEquals(PendingMessage(listOf("first", "second"), block = true), state.pending)
    }

    @Test
    fun anEmptyBlockIsNothing() {
        val (state, actions) = type("\"\"\"", " ", "\"\"\"")
        assertEquals(List(3) { ChatAction.Wait }, actions)
        assertNull(state.pending)
    }

    @Test
    fun aBackslashContinuesTheMessageOnTheNextLine() {
        assertEquals(listOf("one\ntwo\nthree"), sent("one\\", "two\\", "three"))
        // The line after the last backslash ends the message, even an empty one.
        assertEquals(listOf("one", "next"), sent("one\\", "", "next"))
        assertEquals(PendingMessage(listOf("one"), block = false), type("one\\").first.pending)
    }

    @Test
    fun helpListsEveryCommand() {
        val help = said("/help")
        for (command in listOf(
            "/help",
            "/exit",
            "/clear",
            "/model",
            "/system-prompt",
            "/stats",
            "\"\"\"",
        )) {
            assertTrue(command in help, help)
        }
    }

    @Test
    fun exitEndsTheChat() {
        assertEquals(ChatAction.Exit, start.read("/exit").action)
        assertEquals(ChatAction.Exit, start.read("  /exit  ").action)
    }

    @Test
    fun clearForgetsTheConversationAndKeepsTheRest() {
        val talking = start.copy(systemPrompt = "Be brief", showStats = true, previousResponseId = "resp_1")
        val step = talking.read("/clear")
        assertEquals(talking.copy(previousResponseId = null), step.state)
        assertIs<ChatAction.Say>(step.action)
    }

    @Test
    fun modelShowsTheModelOrAsksForAnother() {
        assertEquals("The model is alpha", said("/model"))
        val step = start.read("/model  acme/beta ")
        assertEquals(ChatAction.UseModel("acme/beta"), step.action)
        // Not switched yet: the name may be wrong.
        assertEquals(start, step.state)
    }

    @Test
    fun theSystemPromptIsSetAndRemoved() {
        val set = start.read("/system-prompt  Be brief, please ")
        assertEquals("Be brief, please", set.state.systemPrompt)
        assertEquals("Set the system prompt", assertIs<ChatAction.Say>(set.action).text)

        val removed = set.state.read("/system-prompt")
        assertNull(removed.state.systemPrompt)
        assertEquals("Removed the system prompt", assertIs<ChatAction.Say>(removed.action).text)
        assertEquals("There is no system prompt", said("/system-prompt"))
    }

    @Test
    fun statsShowsTheLastAnswerOrSwitchesThemOnForEveryAnswer() {
        assertEquals("There is no answer yet", said("/stats"))
        val stats =
            ChatStats(inputTokens = 7, totalOutputTokens = 3, tokensPerSecond = 12.5, timeToFirstTokenSeconds = 0.25, stopReason = "length")
        val answered = start.answered(ChatResponse("alpha", emptyList(), stats, "resp_1"))
        assertEquals("resp_1", answered.previousResponseId)
        assertEquals("prompt: 7 tokens, answer: 3 tokens (length), time to first token: 0.25 s, 12.5 tokens/s", said("/stats", answered))

        assertTrue(start.read("/stats on").state.showStats)
        assertEquals(start, start.copy(showStats = true).read("/stats off").state)
        assertIs<ChatAction.Complain>(start.read("/stats maybe").action)
    }

    @Test
    fun whatIsNotACommandIsSaidToBeNone() {
        val step = start.read("/quit now")
        assertEquals(ChatAction.Complain("/quit is not a command; /help lists them"), step.action)
        assertEquals(start, step.state)
    }

    @Test
    fun anAnswerThatWasNotStoredEndsTheConversation() {
        val stats = ChatStats(1, 1, 0.0, 0.0, "stop")
        assertNull(start.copy(previousResponseId = "resp_1").answered(ChatResponse("alpha", emptyList(), stats)).previousResponseId)
    }
}
