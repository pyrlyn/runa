package runa.daemon

import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import kotlinx.serialization.json.JsonPrimitive
import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.ChatFailed
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.ChatStats
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.core.api.LoadRequest
import runa.core.api.ServerRequest
import runa.testing.assertGolden
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** `POST /api/v1/chat`, as a client on a loopback socket sees it. The fake model answers with its prompt. */
class ChatApiTest {
    private suspend fun TestDaemon.chat(
        body: String,
        token: String? = TestDaemon.TOKEN,
        base: String = url,
    ) = post(DaemonRoutes.CHAT, body, token, base)

    /** The answer to [input], which continues the conversation that [previous] ended. */
    private suspend fun TestDaemon.say(
        input: String,
        previous: ChatResponse? = null,
        more: String = "",
    ): ChatResponse {
        val continued = previous?.let { """, "previous_response_id": "${it.responseId}"""" }.orEmpty()
        val response = chat("""{"model": "alpha", "input": "$input"$continued$more}""")
        val text = response.bodyAsText()
        assertEquals(HttpStatusCode.OK, response.status, text)
        return DaemonJson.decodeFromString(ChatResponse.serializer(), text)
    }

    private suspend fun assertRefused(
        response: HttpResponse,
        vararg mentions: String,
    ) {
        val text = response.bodyAsText()
        assertEquals(HttpStatusCode.BadRequest, response.status, text)
        val error = DaemonJson.decodeFromString(ApiError.serializer(), text).error
        assertEquals(ApiError.INVALID_REQUEST, error.type)
        for (mention in mentions) assertTrue(mention in error.message, error.message)
    }

    /** Room for a few turns: the fake model's own context is eight words. */
    private suspend fun TestDaemon.loadAlpha() {
        client().use { it.load(LoadRequest("alpha", contextLength = 64)) }
    }

    @Test
    fun aChatIsAnsweredInOnePiece() =
        withDaemon {
            val response = chat("""{"model": "alpha", "input": "Hello there"}""")
            assertEquals(HttpStatusCode.OK, response.status)
            assertEquals(ContentType.Application.Json, response.contentType()?.withoutParameters())
            assertEquals(
                """{"model_instance_id":"acme/alpha@Q4_K_M","output":[{"type":"message","content":"user: Hello there"}],""" +
                    """"stats":{"input_tokens":3,"total_output_tokens":3,"tokens_per_second":6.67,""" +
                    """"time_to_first_token_seconds":0.01,"stop_reason":"stop"},"response_id":"resp_1"}""",
                response.bodyAsText(),
            )
            // Nobody loaded the model: the request did.
            assertEquals(listOf(ALPHA), registry.instances().map { it.id })
            assertTrue(log.any { it.contains("chat '$ALPHA': 3 prompt tokens, 3 generated") }, "$log")
        }

    @Test
    fun aChatStreamsItsEventsByName() =
        withDaemon {
            val response = chat("""{"model": "alpha", "stream": true, "system_prompt": "Be brief", "input": "Hi"}""")
            assertEquals(HttpStatusCode.OK, response.status)
            assertEquals(ContentType.Text.EventStream, response.contentType()?.withoutParameters())
            assertGolden("native-chat-stream.txt", response.bodyAsText())
        }

    @Test
    fun aRequestContinuesTheConversationItNames() =
        withDaemon {
            loadAlpha()
            val first = say("Hi")
            assertEquals("user: Hi", first.text)
            val second = say("more", first)
            assertEquals("user: Hi assistant: user: Hi user: more", second.text)
            assertEquals(7, second.stats.inputTokens)
            assertEquals(listOf<String?>("resp_1", "resp_2"), listOf(first.responseId, second.responseId))
            // An answer can be continued more than once: the first one still ends where it did.
            assertEquals("user: Hi assistant: user: Hi user: other", say("other", first).text)
        }

    @Test
    fun theSystemPromptStaysUntilARequestGivesAnother() =
        withDaemon {
            loadAlpha()
            val first = say("a", more = """, "system_prompt": "S"""")
            assertEquals("system: S user: a", first.text)
            assertTrue(say("b", first).text.startsWith("system: S user: a assistant: system: S user: a user: b"))
            assertTrue(say("b", first, """, "system_prompt": "T"""").text.startsWith("system: T user: a assistant:"))
            // An empty one removes it.
            assertTrue(say("b", first, """, "system_prompt": """"").text.startsWith("user: a assistant:"))
        }

    @Test
    fun anAnswerThatWasNotStoredCannotBeContinued() =
        withDaemon {
            val answer = say("Hi", more = """, "store": false""")
            assertEquals("user: Hi", answer.text)
            assertNull(answer.responseId)
            assertRefused(chat("""{"model": "alpha", "input": "more", "previous_response_id": "resp_9"}"""), "resp_9")
        }

    @Test
    fun onlyTheConversationsUsedLastAreKept() =
        withDaemon {
            loadAlpha()
            val answers = List(TestDaemon.CONVERSATIONS) { say("turn$it") }
            // Using the oldest one moves it behind the others, and its answer pushes the next oldest out.
            say("again", answers[0])
            assertRefused(chat("""{"model": "alpha", "input": "x", "previous_response_id": "${answers[1].responseId}"}"""), "memory")
            say("again", answers[0])
        }

    @Test
    fun theInputCanComeInParts() =
        withDaemon {
            val response =
                chat("""{"model": "alpha", "input": [{"type": "text", "content": "Hello"}, {"type": "text", "content": "there"}]}""")
            assertEquals("user: Hello\nthere", DaemonJson.decodeFromString(ChatResponse.serializer(), response.bodyAsText()).text)
        }

    @Test
    fun whatIsNotSupportedIsRefusedBeforeAnythingIsLoaded() =
        withDaemon {
            assertRefused(chat("""{"model": "alpha", "input": "Hi", "integrations": [{"type": "plugin"}]}"""), "integrations")
            assertRefused(chat("""{"model": "alpha", "input": [{"type": "image", "data_url": "data:"}]}"""), "image")
            assertRefused(chat("""{"model": "alpha", "input": [{"type": "text"}]}"""), "content")
            assertRefused(chat("""{"model": "alpha", "input": 5}"""), "input")
            assertRefused(chat("""{"model": "alpha", "input": "  "}"""), "input")
            assertRefused(chat("""{"model": "alpha", "input": "Hi", "max_output_tokens": 0}"""), "max_output_tokens")
            assertRefused(chat("""{"model": "alpha", "input": "Hi", "temperature": -1}"""), "temperature")
            assertRefused(chat("""{"model": "alpha"}"""), "input")
            assertRefused(chat("""{"input": "Hi"}"""), "model")
            assertRefused(chat("{not json"))
            val unknown = chat("""{"model": "gamma", "input": "Hi"}""")
            assertEquals(HttpStatusCode.NotFound, unknown.status)
            assertEquals(0, engine.loads.size)
        }

    @Test
    fun aLimitEndsTheAnswerAndSaysSo() =
        withDaemon {
            val answer = say("one two three", more = """, "max_output_tokens": 2""")
            assertEquals("user: one", answer.text)
            assertEquals(ChatStats(4, 2, 5.0, 0.01, "length"), answer.stats)
            // A negative limit is none.
            assertEquals("stop", say("one two three", more = """, "max_output_tokens": -1""").stats.stopReason)
        }

    @Test
    fun aFailureHalfwayIsTheLastEventOfTheStream() =
        withDaemon {
            engine.failOn = "three"
            val stream = chat("""{"model": "alpha", "input": "one two three four", "stream": true}""")
            assertEquals(HttpStatusCode.OK, stream.status)
            val events = stream.bodyAsText().trimEnd().split("\n\n")
            assertEquals(
                listOf("chat.start", "message.start", "message.delta", "message.delta", "message.delta", "error"),
                events.map {
                    it.substringAfter("event: ").substringBefore('\n')
                },
            )
            val failed = DaemonJson.decodeFromString(ChatFailed.serializer(), events.last().substringAfter("data: "))
            assertEquals(ApiError.SERVER_ERROR, failed.error.type)
            assertTrue("three" in failed.error.message)

            val whole = chat("""{"model": "alpha", "input": "one two three four"}""")
            assertEquals(HttpStatusCode.InternalServerError, whole.status)

            // Neither left an answer behind, and the model is free again.
            engine.failOn = null
            assertEquals("resp_1", say("one two three").responseId)
        }

    @Test
    fun theClientGetsThePiecesAndTheWholeAnswer() =
        withDaemon {
            val pieces = mutableListOf<String>()
            val first = client().use { it.chat(ChatRequest("alpha", JsonPrimitive("Hello there")), pieces::add) }
            assertEquals(listOf("user:", " Hello", " there"), pieces)
            assertEquals("user: Hello there", first.text)
            assertEquals(ALPHA, first.modelInstanceId)
            assertEquals("stop", first.stats.stopReason)

            val second = client().use { it.chat(ChatRequest(ALPHA, JsonPrimitive("more"), previousResponseId = first.responseId)) }
            assertTrue(second.text.startsWith("user: Hello there assistant:"), second.text)

            // What is wrong with the request, and what went wrong with the answer, are told apart.
            assertFailsWith<UserInputError> { client().use { it.chat(ChatRequest("gamma", JsonPrimitive("Hi"))) } }
            assertFailsWith<UserInputError> {
                client().use {
                    it.chat(
                        ChatRequest("alpha", JsonPrimitive("Hi"), previousResponseId = "resp_0"),
                    )
                }
            }
            engine.failOn = "three"
            val failure = assertFailsWith<BackendError> { client().use { it.chat(ChatRequest("alpha", JsonPrimitive("one two three"))) } }
            assertTrue("three" in failure.message.orEmpty(), failure.message)
        }

    @Test
    fun theChatNeedsTheDaemonsTokenOnEitherListener() =
        withDaemon {
            val body = """{"model": "alpha", "input": "Hi"}"""
            for (token in listOf(null, "wrong")) assertEquals(HttpStatusCode.Unauthorized, chat(body, token).status)

            val port = freePort()
            client().use { it.server(ServerRequest(running = true, bind = LOOPBACK, port = port, apiKey = "key")) }
            val public = "http://$LOOPBACK:$port"
            // The API key opens the OpenAI-compatible routes and nothing else.
            for (token in listOf(null, "key")) assertEquals(HttpStatusCode.Unauthorized, chat(body, token, public).status)
            assertEquals(0, engine.loads.size)
            assertEquals(HttpStatusCode.OK, chat(body, base = public).status)
        }
}
