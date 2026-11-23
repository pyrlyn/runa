package runa.daemon

import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import runa.core.ToolCall
import runa.core.api.ApiError
import runa.core.api.DaemonJson
import runa.core.api.LoadRequest
import runa.engine.Prompt
import runa.engine.ResponseFormat
import runa.engine.ToolChoice
import runa.store.StoreListing
import runa.testing.assertGolden
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** The OpenAI-compatible routes, as a client on a loopback socket sees them. */
class OpenAiApiTest {
    private suspend fun TestDaemon.chat(body: String) = post(OpenAiRoutes.CHAT_COMPLETIONS, body)

    private val tools =
        """"tools": [{"type": "function", "function": {"name": "lookup", "description": "Finds a word",
        "parameters": {"type": "object", "properties": {"word": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "add"}}]"""

    /** What the engine was last asked. */
    private fun TestDaemon.asked(): Prompt.Chat =
        assertIs<Prompt.Chat>(
            engine.loads
                .single()
                .model.requests
                .last()
                .prompt,
        )

    private suspend fun HttpResponse.error(): ApiError.Detail = DaemonJson.decodeFromString(ApiError.serializer(), bodyAsText()).error

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

    @Test
    fun listsTheStoredModelsAndTheInstancesWithANameOfTheirOwn() =
        withDaemon {
            client().use { it.load(LoadRequest("beta", identifier = "mine")) }
            val response = get(OpenAiRoutes.MODELS)
            assertEquals(HttpStatusCode.OK, response.status)
            assertEquals(
                """{"object":"list","data":[""" +
                    """{"id":"mine","object":"model","created":1700000000,"owned_by":"acme"},""" +
                    """{"id":"acme/alpha@Q4_K_M","object":"model","created":1700000000,"owned_by":"acme"},""" +
                    """{"id":"acme/beta@Q4_K_M","object":"model","created":1700000000,"owned_by":"acme"}]}""",
                response.bodyAsText(),
            )
        }

    @Test
    fun aChatIsAnsweredInOnePiece() =
        withDaemon {
            val response = chat("""{"model": "alpha", "messages": [{"role": "user", "content": "Hello there"}]}""")
            assertEquals(HttpStatusCode.OK, response.status)
            assertEquals(ContentType.Application.Json, response.contentType()?.withoutParameters())
            assertEquals(
                """{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"acme/alpha@Q4_K_M",""" +
                    """"choices":[{"index":0,"message":{"role":"assistant","content":"user: Hello there"},"finish_reason":"stop"}],""" +
                    """"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}}""",
                response.bodyAsText(),
            )
            // Nobody loaded the model: the request did.
            assertEquals(listOf(ALPHA), registry.instances().map { it.id })
            assertTrue(log.any { it.contains("chat.completions '$ALPHA': 3 prompt tokens, 3 generated") }, "$log")
        }

    @Test
    fun aChatStreamsInTheFormatOfOpenAi() =
        withDaemon {
            val response =
                chat(
                    """{"model": "alpha", "stream": true, "stream_options": {"include_usage": true},
                    "messages": [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"}]}""",
                )
            assertEquals(HttpStatusCode.OK, response.status)
            assertEquals(ContentType.Text.EventStream, response.contentType()?.withoutParameters())
            assertGolden("chat-stream.txt", response.bodyAsText())
        }

    @Test
    fun aStreamThatHitsItsLimitSaysSo() =
        withDaemon {
            val response =
                chat("""{"model": "alpha", "stream": true, "max_tokens": 2, "messages": [{"role": "user", "content": "one two three"}]}""")
            assertGolden("chat-stream-length.txt", response.bodyAsText())
        }

    @Test
    fun aModelCallsATool() =
        withDaemon {
            val response = chat("""{"model": "alpha", $tools, "messages": [{"role": "user", "content": "please lookup this"}]}""")
            assertEquals(
                """{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"acme/alpha@Q4_K_M",""" +
                    """"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[""" +
                    """{"id":"call_0","type":"function","function":{"name":"lookup","arguments":"{\"words\":4}"}}]},""" +
                    """"finish_reason":"tool_calls"}],""" +
                    """"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}""",
                response.bodyAsText(),
            )
            val asked = asked()
            assertEquals(listOf("lookup", "add"), asked.tools.map { it.name })
            assertEquals(listOf("Finds a word", null), asked.tools.map { it.description })
            assertEquals(
                Json.parseToJsonElement("""{"type":"object","properties":{"word":{"type":"string"}}}""").jsonObject,
                asked.tools.first().parameters,
            )
            assertEquals(ToolChoice.Auto, asked.toolChoice)
            assertEquals(null, asked.parallelToolCalls)
        }

    @Test
    fun toolCallsStreamPieceByPiece() =
        withDaemon {
            val response =
                chat(
                    """{"model": "alpha", "stream": true, "stream_options": {"include_usage": true}, $tools,
                    "messages": [{"role": "user", "content": "add then lookup"}]}""",
                )
            assertEquals(HttpStatusCode.OK, response.status)
            assertGolden("chat-stream-tools.txt", response.bodyAsText())
        }

    @Test
    fun aToolChoiceAndAResponseFormatReachTheModel() =
        withDaemon {
            suspend fun answer(fields: String): String {
                val response = chat("""{"model": "alpha", "messages": [{"role": "user", "content": "Hi"}], $fields}""")
                val text = response.bodyAsText()
                assertEquals(HttpStatusCode.OK, response.status, text)
                return text
            }
            assertTrue(""""name":"add"""" in answer("""$tools, "tool_choice": {"type": "function", "function": {"name": "add"}}"""))
            assertEquals(ToolChoice.Named("add"), asked().toolChoice)
            assertTrue(""""name":"lookup"""" in answer("""$tools, "tool_choice": "required", "parallel_tool_calls": false"""))
            assertEquals(ToolChoice.Required, asked().toolChoice)
            assertEquals(false, asked().parallelToolCalls)
            assertTrue(""""content":"user: Hi"""" in answer("""$tools, "tool_choice": "none""""))
            assertEquals(ToolChoice.None, asked().toolChoice)

            assertTrue(""""content":"{\"echo\":\"user: Hi\"}"""" in answer(""""response_format": {"type": "json_object"}"""))
            assertEquals(ResponseFormat.Json(), asked().format)
            val schema = """{"type":"object","properties":{"word":{"type":"string"}}}"""
            answer(""""response_format": {"type": "json_schema", "json_schema": {"name": "word", "strict": true, "schema": $schema}}""")
            assertEquals(ResponseFormat.Json(Json.parseToJsonElement(schema).jsonObject), asked().format)
            // llama.cpp's own way to send a schema.
            answer(""""response_format": {"type": "json_object", "schema": $schema}""")
            assertEquals(ResponseFormat.Json(Json.parseToJsonElement(schema).jsonObject), asked().format)
        }

    @Test
    fun aCompletionContinuesThePromptAsItIs() =
        withDaemon {
            val whole = post(OpenAiRoutes.COMPLETIONS, """{"model": "alpha", "prompt": "Once upon a time", "max_tokens": 3}""")
            assertEquals(
                """{"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"acme/alpha@Q4_K_M",""" +
                    """"choices":[{"index":0,"text":"Once upon a","finish_reason":"length"}],""" +
                    """"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}""",
                whole.bodyAsText(),
            )
            val stream =
                post(
                    OpenAiRoutes.COMPLETIONS,
                    """{"model": "alpha", "prompt": ["Once upon"], "stream": true, "stream_options": {"include_usage": true}}""",
                )
            assertGolden("completion-stream.txt", stream.bodyAsText())
        }

    @Test
    fun aRequestIsReadTheWayClientsWriteIt() =
        withDaemon {
            suspend fun answer(body: String): String {
                val response = chat(body)
                val text = response.bodyAsText()
                assertEquals(HttpStatusCode.OK, response.status, text)
                return text.substringAfter(""""content":"""").substringBefore('"')
            }
            // Content in parts, the newer name of the system role, and one stop string.
            assertEquals(
                "system: Be brief user: Hello there",
                answer(
                    """{"model": "alpha", "stop": "END", "messages": [
                    {"role": "developer", "content": "Be brief"},
                    {"role": "user", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there END more"}]}]}""",
                ),
            )
            // Several stop strings, and the newer name of the limit, which wins over the older one.
            assertEquals(
                "user: one",
                answer(
                    """{"model": "alpha", "stop": ["nine", "three"], "max_completion_tokens": 2, "max_tokens": 9,
                    "messages": [{"role": "user", "content": "one two three"}]}""",
                ),
            )
            // An explicit null is a default, a negative limit is none, and fields nobody knows are skipped.
            assertEquals(
                "user: one two assistant: tool: 42",
                answer(
                    """{"model": "alpha", "stream": null, "stop": null, "temperature": null, "max_tokens": -1, "n": 1, "tools": [],
                    "response_format": {"type": "text"}, "user": "someone", "frequency_penalty": 0,
                    "messages": [{"role": "user", "content": "one two"},
                    {"role": "assistant", "content": null, "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": "{\"a\":40}"}}]},
                    {"role": "tool", "content": "42", "tool_call_id": "call_1"}]}""",
                ),
            )
            val (_, called, result) = asked().messages
            assertEquals(listOf(ToolCall("call_1", "add", """{"a":40}""")), called.toolCalls)
            assertEquals("call_1", result.toolCallId)
            assertEquals(1, engine.loads.size)
        }

    @Test
    fun whatIsNotSupportedIsRefusedBeforeAnythingIsLoaded() =
        withDaemon {
            val user = """"messages": [{"role": "user", "content": "Hi"}]"""
            assertRefused(chat("""{"model": "alpha", $user, "tools": [{"type": "function"}]}"""), "function")
            assertRefused(chat("""{"model": "alpha", $user, "tools": [{"type": "retrieval"}]}"""), "retrieval")
            assertRefused(chat("""{"model": "alpha", $user, "functions": [{"name": "add"}]}"""), "functions")
            assertRefused(chat("""{"model": "alpha", $user, $tools, "tool_choice": "sometimes"}"""), "tool_choice")
            assertRefused(
                chat("""{"model": "alpha", $user, $tools, "tool_choice": {"type": "function", "function": {"name": "divide"}}}"""),
                "divide",
            )
            assertRefused(chat("""{"model": "alpha", $user, "tool_choice": "required"}"""), "no tools")
            assertRefused(chat("""{"model": "alpha", $user, "response_format": {"type": "xml"}}"""), "xml")
            assertRefused(chat("""{"model": "alpha", $user, "response_format": {"type": "json_schema", "json_schema": {}}}"""), "schema")
            assertRefused(chat("""{"model": "alpha", $user, "response_format": "json"}"""), "response_format")
            assertRefused(chat("""{"model": "alpha", $user, "n": 2}"""), "n must be 1")
            assertRefused(chat("""{"model": "alpha", $user, "max_tokens": 0}"""), "max_tokens")
            assertRefused(chat("""{"model": "alpha", $user, "temperature": -1}"""), "temperature")
            assertRefused(chat("""{"model": "alpha", $user, "stop": 5}"""), "stop")
            assertRefused(chat("""{"model": "alpha", "messages": []}"""), "messages")
            assertRefused(chat("""{"model": "alpha", "messages": [{"role": "narrator", "content": "Hi"}]}"""), "narrator")
            assertRefused(
                chat("""{"model": "alpha", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}"""),
                "image_url",
            )
            assertRefused(chat("""{"messages": []}"""), "model")
            assertRefused(chat("{not json"))
            assertRefused(post(OpenAiRoutes.COMPLETIONS, """{"model": "alpha", "prompt": ["one", "two"]}"""), "prompt")
            assertRefused(post(OpenAiRoutes.COMPLETIONS, """{"model": "alpha", "prompt": "one", "echo": true}"""), "echo")
            assertEquals(0, engine.loads.size)
        }

    @Test
    fun aModelNobodyHasIsNotFound() =
        withDaemon(StoreListing(listOf(stored("alpha"), stored("alpha", "Q8_0")), emptyList())) {
            val unknown = chat("""{"model": "gamma", "messages": [{"role": "user", "content": "Hi"}]}""")
            assertEquals(HttpStatusCode.NotFound, unknown.status)
            val error = unknown.error()
            assertEquals(ApiError.MODEL_NOT_FOUND, error.code)
            assertEquals(ApiError.INVALID_REQUEST, error.type)
            assertTrue("gamma" in error.message)
            // A name that fits two models is a mistake of another kind: the answer names them.
            assertRefused(chat("""{"model": "alpha", "messages": [{"role": "user", "content": "Hi"}]}"""), ALPHA, "acme/alpha@Q8_0")
            assertEquals(0, engine.loads.size)
        }

    @Test
    fun requestsForAModelThatIsNotLoadedLoadItOnce() =
        withDaemon {
            val body = """{"model": "acme/alpha", "messages": [{"role": "user", "content": "Hi"}]}"""
            val answers = coroutineScope { List(4) { async { chat(body).status } }.awaitAll() }
            assertEquals(List(4) { HttpStatusCode.OK }, answers)
            assertEquals(1, engine.loads.size)
        }

    @Test
    fun aPromptTheModelCannotTakeGetsAStatusEvenWhenStreaming() =
        withDaemon {
            assertRefused(post(OpenAiRoutes.COMPLETIONS, """{"model": "alpha", "prompt": "", "stream": true}"""), "prompt is empty")
        }

    @Test
    fun aFailureHalfwayIsTheLastEventOfTheStream() =
        withDaemon {
            engine.failOn = "three"
            val stream = post(OpenAiRoutes.COMPLETIONS, """{"model": "alpha", "prompt": "one two three four", "stream": true}""")
            assertEquals(HttpStatusCode.OK, stream.status)
            val events = stream.bodyAsText().trimEnd().split("\n\n")
            assertEquals(3, events.size, "$events")
            assertTrue(""""text":" two"""" in events[1])
            val error = DaemonJson.decodeFromString(ApiError.serializer(), events[2].removePrefix("data: ")).error
            assertEquals(ApiError.SERVER_ERROR, error.type)
            assertTrue("three" in error.message)

            val whole = post(OpenAiRoutes.COMPLETIONS, """{"model": "alpha", "prompt": "one two three four"}""")
            assertEquals(HttpStatusCode.InternalServerError, whole.status)
            assertEquals(ApiError.SERVER_ERROR, whole.error().type)

            // The model is free again after both.
            engine.failOn = null
            assertEquals(HttpStatusCode.OK, post(OpenAiRoutes.COMPLETIONS, """{"model": "alpha", "prompt": "one two three"}""").status)
        }

    @Test
    fun embeddingsComeAsNumbersOrAsBase64() =
        withDaemon {
            val floats = post(OpenAiRoutes.EMBEDDINGS, """{"model": "beta", "input": ["ab", "four"]}""")
            assertEquals(
                """{"object":"list","data":[{"object":"embedding","index":0,"embedding":[2.0]},""" +
                    """{"object":"embedding","index":1,"embedding":[4.0]}],""" +
                    """"model":"acme/beta@Q4_K_M","usage":{"prompt_tokens":2,"total_tokens":2}}""",
                floats.bodyAsText(),
            )
            // Little-endian 32-bit floats: 2.0 is 00 00 00 40.
            val packed = post(OpenAiRoutes.EMBEDDINGS, """{"model": "beta", "input": "ab", "encoding_format": "base64"}""")
            assertTrue(""""embedding":"AAAAQA=="""" in packed.bodyAsText(), packed.bodyAsText())

            assertRefused(post(OpenAiRoutes.EMBEDDINGS, """{"model": "beta", "input": []}"""), "input")
            assertRefused(post(OpenAiRoutes.EMBEDDINGS, """{"model": "beta", "input": [[1, 2, 3]]}"""), "input")
            assertRefused(post(OpenAiRoutes.EMBEDDINGS, """{"model": "beta", "input": "ab", "dimensions": 8}"""), "dimensions")
            assertRefused(
                post(OpenAiRoutes.EMBEDDINGS, """{"model": "beta", "input": "ab", "encoding_format": "hex"}"""),
                "encoding_format",
            )
        }

    @Test
    fun onTheDaemonsOwnPortTheRoutesNeedItsToken() =
        withDaemon {
            for (token in listOf(null, "wrong")) {
                val models = get(OpenAiRoutes.MODELS, token)
                assertEquals(HttpStatusCode.Unauthorized, models.status)
                val error = models.error()
                assertEquals(ApiError.UNAUTHORIZED, error.type)
                assertEquals(ApiError.INVALID_API_KEY, error.code)
                val chat =
                    post(OpenAiRoutes.CHAT_COMPLETIONS, """{"model": "alpha", "messages": [{"role": "user", "content": "Hi"}]}""", token)
                assertEquals(HttpStatusCode.Unauthorized, chat.status)
            }
            assertEquals(0, engine.loads.size)
            assertEquals(HttpStatusCode.OK, get(OpenAiRoutes.MODELS).status)
        }
}
