package runa.client

import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.toByteArray
import io.ktor.client.request.HttpResponseData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
import okio.IOException
import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.UnloadRequest
import runa.testing.fixture
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

/**
 * The contract with LM Studio's v1 REST API: every answer is one of `fixtures/lmstudio`, which are
 * the examples of its documentation and, for the failures it does not document, what LM Studio
 * 0.4.28 answered.
 */
class LmStudioBackendTest {
    /** `POST /api/v1/models/load {...}`, in order. */
    private val requests = mutableListOf<String>()
    private val tokens = mutableListOf<String?>()

    private fun recorded(name: String) = fixture("lmstudio/$name")

    private fun MockRequestHandleScope.json(
        name: String,
        status: HttpStatusCode = HttpStatusCode.OK,
    ) = respond(recorded(name), status, headersOf(HttpHeaders.ContentType, "application/json"))

    private fun MockRequestHandleScope.events(name: String) =
        respond(recorded(name), HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "text/event-stream"))

    private fun <T> lmStudio(
        token: String? = null,
        answer: MockRequestHandleScope.(path: String) -> HttpResponseData,
        block: suspend LmStudioBackend.() -> T,
    ): T {
        val engine =
            MockEngine { request ->
                requests += "${request.method.value} ${request.url.encodedPath} ${request.body.toByteArray().decodeToString()}".trim()
                tokens += request.headers[HttpHeaders.Authorization]
                answer(request.url.encodedPath)
            }
        return LmStudioBackend(engine, URL, token).use { runBlocking { it.block() } }
    }

    /**
     * The documented list has `google/gemma-4-26b-a4b` loaded and the documented load answers for
     * `openai/gpt-oss-20b`: the list after that load is the documented one under the other name.
     */
    private fun MockRequestHandleScope.afterTheLoad() =
        respond(recorded("models.json").replace(GEMMA, GPT_OSS), HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "application/json"))

    @Test
    fun listsTheModelsInRunasOwnTypes() {
        val models = lmStudio(answer = { json("models.json") }) { models().models }

        assertEquals(listOf("GET /api/v1/models"), requests)
        assertEquals(listOf(GEMMA, "deepseek-r1", "text-embedding-nomic-embed-text-v1.5-embedding"), models.map { it.key })
        assertEquals(listOf("llm", "llm", "embedding"), models.map { it.type })
        assertEquals(listOf("Q4_K_M", "Q4_K_M", "F16"), models.map { it.quantization })
        assertEquals(listOf("gemma4", "deepseek", null), models.map { it.architecture })
        // "26B-A4B" is not one count; a missing label is none.
        assertEquals(listOf<Long?>(null, 671_000_000_000, null), models.map { it.params })
        assertEquals(listOf(17_990_911_801, 40_492_610_355, 274_290_560), models.map { it.sizeBytes })
        assertEquals(listOf<Long?>(262_144, 131_072, 2_048), models.map { it.maxContextLength })
        assertEquals("Gemma 4 26B A4B", models.first().displayName)
    }

    @Test
    fun leavesNullWhatLmStudioDoesNotTellAboutAnInstance() {
        val instance =
            lmStudio(answer = { json("models.json") }) { models().models }
                .flatMap { it.loadedInstances }
                .single()

        assertEquals(GEMMA, instance.id)
        assertEquals(GEMMA, instance.model)
        assertEquals(4_096, instance.contextLength)
        assertEquals(4, instance.parallel)
        assertEquals(17_990_911_801, instance.sizeBytes)
        assertNull(instance.gpu)
        assertNull(instance.ttlSeconds)
        assertNull(instance.idleSeconds)
    }

    @Test
    fun sendsTheApiTokenAsABearerAndNoneWithoutOne() {
        lmStudio(token = "lm-token", answer = { json("models.json") }) { models() }
        lmStudio(answer = { json("models.json") }) { models() }

        assertEquals(listOf("Bearer lm-token", null), tokens)
    }

    @Test
    fun loadsAModelWithItsContextLengthAndNothingElse() {
        val response =
            lmStudio(answer = { path -> if (path == LOAD) json("load.json") else afterTheLoad() }) {
                load(LoadRequest(GPT_OSS, contextLength = 16_384, stream = true))
            }

        // A key LM Studio does not know, or a null, makes it refuse the request.
        assertEquals("""POST $LOAD {"model":"$GPT_OSS","context_length":16384}""", requests[1])
        assertEquals(LoadResponse.LOADED, response.status)
        assertEquals(9.099, response.loadTimeSeconds)
        assertNull(response.estimate)
        assertEquals(GPT_OSS, response.instance?.id)
        assertEquals(4_096, response.instance?.contextLength)
    }

    @Test
    fun loadsWithLmStudiosOwnContextLengthWhenNoneIsAsked() {
        lmStudio(answer = { path -> if (path == LOAD) json("load.json") else afterTheLoad() }) { load(LoadRequest(GPT_OSS)) }

        assertEquals("""POST $LOAD {"model":"$GPT_OSS"}""", requests[1])
    }

    @Test
    fun loadsAModelNamedWithoutItsPublisher() {
        lmStudio(answer = { path -> if (path == LOAD) json("load.json") else afterTheLoad() }) { load(LoadRequest("GPT-OSS-20B")) }

        assertEquals("""POST $LOAD {"model":"$GPT_OSS"}""", requests[1])
    }

    @Test
    fun refusesTheLoadSettingsLmStudiosApiDoesNotHave() {
        val error =
            assertFailsWith<UserInputError> {
                lmStudio(answer = { error("nothing is asked") }) { load(LoadRequest(GPT_OSS, gpu = "max", ttl = 60, parallel = 2)) }
            }

        assertContains(error.message.orEmpty(), "it cannot take gpu, parallel, ttl")
        assertEquals(emptyList(), requests)
    }

    @Test
    fun aLoadedModelThatIsNotListedIsAFailure() {
        val error =
            assertFailsWith<BackendError> {
                lmStudio(answer = { path -> if (path == LOAD) json("load.json") else json("models.json") }) { load(LoadRequest(GPT_OSS)) }
            }

        assertEquals("LM Studio loaded '$GPT_OSS' and does not list it", error.message)
    }

    @Test
    fun unloadsOneInstance() {
        val response = lmStudio(answer = { json("unload.json") }) { unload(UnloadRequest(instanceId = GPT_OSS)) }

        assertEquals(listOf("""POST $UNLOAD {"instance_id":"$GPT_OSS"}"""), requests)
        assertEquals(listOf(GPT_OSS), response.unloaded)
    }

    @Test
    fun unloadsEveryListedInstanceForAll() {
        val response =
            lmStudio(answer = { path -> if (path == UNLOAD) json("unload.json") else json("models.json") }) {
                unload(UnloadRequest(all = true))
            }

        assertEquals(listOf("GET /api/v1/models", """POST $UNLOAD {"instance_id":"$GEMMA"}"""), requests)
        assertEquals(1, response.unloaded.size)
    }

    @Test
    fun streamsTheChatAndReturnsItsLastEvent() {
        val pieces = mutableListOf<String>()
        val response =
            lmStudio(answer = { events("chat-stream.txt") }) {
                chat(ChatRequest(GPT_OSS, JsonPrimitive("What is trending?"), systemPrompt = "", store = false)) { pieces += it }
            }

        assertEquals(
            listOf("""POST /api/v1/chat {"model":"$GPT_OSS","input":"What is trending?","system_prompt":"","stream":true,"store":false}"""),
            requests,
        )
        // Only the answer is text for the user: the reasoning and the tool calls are not printed.
        assertEquals(listOf("The current"), pieces)
        assertEquals(GPT_OSS, response.modelInstanceId)
        assertEquals(listOf("reasoning", "tool_call", ChatResponse.MESSAGE), response.output.map { it.type })
        assertEquals("The current top‑trending model is...", response.text)
        assertEquals(329, response.stats.inputTokens)
        assertEquals(268, response.stats.totalOutputTokens)
        assertEquals(43.73, response.stats.tokensPerSecond)
        assertEquals(0.781, response.stats.timeToFirstTokenSeconds)
        assertNull(response.stats.stopReason)
        assertEquals("resp_02b2017dbc06c12bfc353a2ed6c2b802f8cc682884bb5716", response.responseId)
    }

    @Test
    fun continuesAConversationByTheIdOfItsLastAnswer() {
        lmStudio(answer = { events("chat-stream.txt") }) {
            chat(ChatRequest(GPT_OSS, JsonPrimitive("And then?"), previousResponseId = "resp_1"))
        }

        assertEquals(
            """POST /api/v1/chat {"model":"$GPT_OSS","input":"And then?","stream":true,"previous_response_id":"resp_1"}""",
            requests.single(),
        )
    }

    @Test
    fun anErrorEventEndsTheChat() {
        val error =
            assertFailsWith<UserInputError> {
                lmStudio(answer = { events("chat-stream-error.txt") }) { chat(ChatRequest(GPT_OSS, JsonPrimitive("hi"))) }
            }

        assertEquals("\"model\" is required", error.message)
    }

    @Test
    fun aStreamWithoutItsLastEventIsAFailure() {
        val cut = recorded("chat-stream.txt").substringBefore("event: chat.end")
        val error =
            assertFailsWith<BackendError> {
                lmStudio(answer = { respond(cut, HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "text/event-stream")) }) {
                    chat(ChatRequest(GPT_OSS, JsonPrimitive("hi")))
                }
            }

        assertEquals("LM Studio closed the connection before the answer was complete", error.message)
    }

    @Test
    fun readsAnAnswerWithToolCalls() {
        val response = LmStudioJson.decodeFromString(ChatResponse.serializer(), recorded("chat.json"))

        assertEquals(listOf("tool_call", "message", "tool_call", "message"), response.output.map { it.type })
        assertEquals("", response.output.first().content)
        assertContains(response.text, "Top Trending Model on Hugging Face")
        assertEquals(646, response.stats.inputTokens)
        assertEquals(29.75, response.stats.tokensPerSecond)
    }

    @Test
    fun startsADownload() {
        val job = lmStudio(answer = { json("download.json") }) { download("ibm/granite-4-micro") }

        assertEquals(listOf("""POST /api/v1/models/download {"model":"ibm/granite-4-micro"}"""), requests)
        assertEquals("job_493c7c9ded", job.jobId)
        assertEquals(LmStudioDownload.DOWNLOADING, job.status)
        assertEquals(2_279_145_003, job.totalSizeBytes)
        assertNull(job.downloadedBytes)
    }

    @Test
    fun namesTheQuantizationOfADownloadFromHuggingFace() {
        lmStudio(answer = { json("download.json") }) { download("https://huggingface.co/acme/tiny-GGUF", "Q4_K_M") }

        assertEquals(
            """POST /api/v1/models/download {"model":"https://huggingface.co/acme/tiny-GGUF","quantization":"Q4_K_M"}""",
            requests.single(),
        )
    }

    @Test
    fun asksHowFarADownloadIs() {
        val job = lmStudio(answer = { json("download-status.json") }) { downloadStatus("job_493c7c9ded") }

        assertEquals(listOf("GET /api/v1/models/download/status/job_493c7c9ded"), requests)
        assertEquals(LmStudioDownload.COMPLETED, job.status)
        assertEquals(2_279_145_003, job.downloadedBytes)
        assertEquals("2025-10-03T15:43:12.102Z", job.completedAt)
    }

    @Test
    fun aModelLmStudioDoesNotHaveIsTheUsersMistake() {
        val notLoaded =
            assertFailsWith<UserInputError> {
                lmStudio(answer = { json("error-not-loaded.json", HttpStatusCode.NotFound) }) { unload(UnloadRequest("no-such-instance")) }
            }
        val notDownloaded =
            assertFailsWith<UserInputError> {
                lmStudio(answer = { path ->
                    if (path == LOAD) json("error-not-downloaded.json", HttpStatusCode.NotFound) else json("models.json")
                }) { load(LoadRequest("no/such-model")) }
            }

        assertEquals("Model with instance identifier 'no-such-instance' is not loaded.", notLoaded.message)
        assertEquals("Model no/such-model not found in downloaded models", notDownloaded.message)
    }

    @Test
    fun aRequestLmStudioRefusesIsTheUsersMistake() {
        val error =
            assertFailsWith<UserInputError> {
                lmStudio(answer = { json("error-unrecognized-key.json", HttpStatusCode.BadRequest) }) { download("x") }
            }

        assertEquals("Unrecognized key(s) in object: 'gpu'", error.message)
    }

    @Test
    fun aPathLmStudioDoesNotServeIsAFailure() {
        val error =
            assertFailsWith<BackendError> {
                lmStudio(answer = { json("error-no-such-path.json", HttpStatusCode.NotFound) }) { models() }
            }

        assertEquals("LM Studio at $URL answered 404: Unexpected endpoint or method. (GET /api/v1/status)", error.message)
    }

    @Test
    fun aRejectedTokenSaysWhereToGiveTheRightOne() {
        val error =
            assertFailsWith<BackendError> {
                lmStudio(answer = { respond("", HttpStatusCode.Unauthorized) }) { models() }
            }

        assertEquals("LM Studio at $URL rejected the API token; give the right one with --token or LM_API_TOKEN", error.message)
    }

    @Test
    fun anAnswerThatIsNotLmStudiosSaysSo() {
        val error =
            assertFailsWith<BackendError> {
                lmStudio(answer = { respond("<html>hello</html>") }) { models() }
            }

        assertEquals("$URL sent an answer that runa cannot read; is it LM Studio 0.4.0 or newer?", error.message)
    }

    @Test
    fun anLmStudioThatDoesNotRunSaysSo() {
        val error =
            assertFailsWith<BackendError> {
                lmStudio(answer = { throw IOException("Connection refused") }) { models() }
            }

        assertEquals("Cannot reach LM Studio at $URL: Connection refused", error.message)
    }

    @Test
    fun theReasonOfAFailedConnectionStaysShort() {
        val reason = """Exception in http request: Error Domain=NSURLErrorDomain Code=-1004 "Could not connect to the server.""""
        val error =
            assertFailsWith<BackendError> {
                lmStudio(answer = { throw IOException("$reason UserInfo={_kCFStreamErrorCodeKey=61, NSUnderlyingError=0x1}") }) { models() }
            }

        assertEquals("Cannot reach LM Studio at $URL: $reason", error.message)
    }

    @Test
    fun readsAParameterCountFromItsLabel() {
        assertEquals(7_000_000_000, LmStudioBackend.parameterCount("7B"))
        assertEquals(1_500_000_000, LmStudioBackend.parameterCount("1.5B"))
        assertEquals(270_000_000, LmStudioBackend.parameterCount("270m"))
        assertNull(LmStudioBackend.parameterCount("26B-A4B"))
        assertNull(LmStudioBackend.parameterCount(null))
    }

    private companion object {
        const val URL = "http://127.0.0.1:1234"
        const val GEMMA = "google/gemma-4-26b-a4b"
        const val GPT_OSS = "openai/gpt-oss-20b"
        const val LOAD = "/api/v1/models/load"
        const val UNLOAD = "/api/v1/models/unload"
    }
}
