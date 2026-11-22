package runa.engine.llama

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import runa.core.ChatMessage
import runa.core.ChatRole
import runa.core.ToolCall
import runa.core.UserInputError
import runa.engine.GenerationEvent
import runa.engine.Prompt
import runa.engine.ResponseFormat
import runa.engine.Tool
import runa.engine.ToolChoice
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ChatTurnTest {
    private val user = ChatMessage(ChatRole.USER, "What is 2 + 2?")
    private val numbers = Json.parseToJsonElement("""{"type":"object","properties":{"a":{"type":"number"}}}""").jsonObject
    private val add = Tool("add", "Adds numbers", numbers)

    @Test
    fun aPlainChatIsOnlyItsMessages() {
        assertEquals("""{"messages":[{"role":"user","content":"What is 2 + 2?"}]}""", chatRequest(Prompt.Chat(listOf(user))).toString())
        assertFailsWith<UserInputError> { chatRequest(Prompt.Chat(emptyList())) }
    }

    @Test
    fun toolsAndTheirCallsAreWrittenAsOpenAiWritesThem() {
        val messages =
            listOf(
                user,
                ChatMessage(ChatRole.ASSISTANT, "", toolCalls = listOf(ToolCall("call_1", "add", """{"a":2}"""))),
                ChatMessage(ChatRole.TOOL, "4", toolCallId = "call_1"),
            )
        val chat = Prompt.Chat(messages, listOf(add, Tool("now")), parallelToolCalls = false)
        assertEquals(
            """{"messages":[{"role":"user","content":"What is 2 + 2?"},""" +
                """{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function",""" +
                """"function":{"name":"add","arguments":"{\"a\":2}"}}]},""" +
                """{"role":"tool","content":"4","tool_call_id":"call_1"}],""" +
                """"tools":[{"type":"function","function":{"name":"add","description":"Adds numbers","parameters":$numbers}},""" +
                """{"type":"function","function":{"name":"now"}}],"tool_choice":"auto","parallel_tool_calls":false}""",
            chatRequest(chat).toString(),
        )
    }

    @Test
    fun aNamedToolIsTheOnlyOneOfferedAndRequired() {
        val request = chatRequest(Prompt.Chat(listOf(user), listOf(add, Tool("now")), ToolChoice.Named("now"))).toString()
        assertEquals(
            """{"messages":[{"role":"user","content":"What is 2 + 2?"}],""" +
                """"tools":[{"type":"function","function":{"name":"now"}}],"tool_choice":"required"}""",
            request,
        )
        assertFailsWith<UserInputError> { chatRequest(Prompt.Chat(listOf(user), listOf(add), ToolChoice.Named("now"))) }
    }

    @Test
    fun toolsThatMayNotBeCalledAreNotShown() {
        val request = chatRequest(Prompt.Chat(listOf(user), listOf(add), ToolChoice.None, parallelToolCalls = true))
        assertEquals("""{"messages":[{"role":"user","content":"What is 2 + 2?"}]}""", request.toString())
    }

    @Test
    fun aJsonFormatIsASchema() {
        fun schema(format: ResponseFormat) = chatRequest(Prompt.Chat(listOf(user), format = format))["json_schema"].toString()
        assertEquals("""{"type":"object"}""", schema(ResponseFormat.Json()))
        assertEquals(numbers.toString(), schema(ResponseFormat.Json(numbers)))
        assertEquals("null", schema(ResponseFormat.Text))
    }

    @Test
    fun readsWhatTheShimReports() {
        val start = turnStart("""{"prompt":"<|im_start|>user\nHi<|im_end|>\n","stops":["<|im_end|>"]}""")
        assertEquals("<|im_start|>user\nHi<|im_end|>\n", start.prompt)
        assertEquals(listOf("<|im_end|>"), start.stops)

        assertEquals(emptyList(), turnEvents("[]"))
        assertEquals(
            listOf(
                GenerationEvent.Text("Let me add."),
                GenerationEvent.ToolCallStarted(0, "call_x", "add"),
                GenerationEvent.ToolCallArguments(0, """{"a":"""),
            ),
            turnEvents(
                """[{"content":"Let me add."},{"call":{"index":0,"id":"call_x","name":"add"}},{"arguments":{"index":0,"text":"{\"a\":"}}]""",
            ),
        )
    }
}
