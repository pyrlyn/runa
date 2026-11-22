package runa.cli

import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.toByteArray
import io.ktor.client.request.HttpRequestData
import io.ktor.client.request.HttpResponseData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.serialization.SerializationStrategy
import kotlinx.serialization.json.JsonPrimitive
import okio.FileSystem
import okio.IOException
import okio.Path
import runa.config.DaemonInfo
import runa.config.DiscoveryFile
import runa.core.api.ApiError
import runa.core.api.ChatDelta
import runa.core.api.ChatEnded
import runa.core.api.ChatFailed
import runa.core.api.ChatMark
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.ChatStarted
import runa.core.api.ChatStats
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.core.api.DaemonStatus
import runa.core.api.EngineDescription
import runa.core.api.InstanceDescription
import runa.core.api.LoadProgress
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.LogEvent
import runa.core.api.MemoryEstimateDescription
import runa.core.api.ModelDescription
import runa.core.api.ModelList
import runa.core.api.ServerDescription
import runa.core.api.ServerRequest
import runa.core.api.ShutdownResponse
import runa.core.api.UnloadRequest
import runa.core.api.UnloadResponse

/**
 * What the CLI sees of a daemon, without a process: the HTTP API, the discovery file it writes
 * when started, and the lock that says it is alive. The real daemon is tested in `:daemon` and
 * through the executable in `:app`.
 */
internal class FakeDaemon(
    private val home: Path,
    private val version: String = "1.2.3",
    /** False for a daemon that is started and never comes up. */
    private val comesUp: Boolean = true,
) {
    var running = false
        private set

    /** Where it listens: the port it was started with, as the real one does. */
    var port = PORT
        private set

    /** The arguments of every start, in order. */
    val starts = mutableListOf<List<String>>()

    /** Every request that passed the token check: `POST /api/v1/models/load {...}`. */
    val requests = mutableListOf<String>()
    val instances = mutableListOf<InstanceDescription>()

    /** How many answers it stored, which is what their ids count. */
    private var answers = 0

    /** What `server start` opened. */
    var server = ServerDescription()
        private set

    val discovery = DiscoveryFile(FileSystem.SYSTEM, home / "run" / "daemon.json")

    val http: HttpClientEngine = MockEngine { request -> answer(request) }

    fun start(arguments: List<String> = emptyList()) {
        starts += arguments
        if (!comesUp) return
        running = true
        arguments.indexOf("--port").takeIf { it >= 0 }?.let { port = arguments[it + 1].toInt() }
        discovery.write(DaemonInfo(PID, port, TOKEN, version))
    }

    /** What `kill -9` leaves behind: no process, but its discovery file. */
    fun kill() {
        running = false
    }

    private suspend fun MockRequestHandleScope.answer(request: HttpRequestData): HttpResponseData {
        if (!running || request.url.port != port) throw IOException("Connection refused")
        val path = request.url.encodedPath
        if (path == DaemonRoutes.STATUS) return json(DaemonStatus.serializer(), status())
        if (request.headers[HttpHeaders.Authorization] != "Bearer $TOKEN") {
            return json(
                ApiError.serializer(),
                ApiError(ApiError.Detail("A valid bearer token is required", ApiError.UNAUTHORIZED)),
                HttpStatusCode.Unauthorized,
            )
        }
        val body = request.body.toByteArray().decodeToString()
        requests += "${request.method.value} ${request.url.encodedPathAndQuery} $body".trim()
        return when (path) {
            DaemonRoutes.MODELS -> {
                json(
                    ModelList.serializer(),
                    ModelList(
                        MODELS.map { model ->
                            model.copy(
                                loadedInstances =
                                    instances.filter {
                                        it.model ==
                                            model.key
                                    },
                            )
                        },
                    ),
                )
            }

            DaemonRoutes.LOAD -> {
                load(DaemonJson.decodeFromString(LoadRequest.serializer(), body))
            }

            DaemonRoutes.UNLOAD -> {
                unload(DaemonJson.decodeFromString(UnloadRequest.serializer(), body))
            }

            DaemonRoutes.CHAT -> {
                chat(DaemonJson.decodeFromString(ChatRequest.serializer(), body))
            }

            DaemonRoutes.SERVER -> {
                serve(DaemonJson.decodeFromString(ServerRequest.serializer(), body))
            }

            DaemonRoutes.LOGS -> {
                logs(request.url.parameters[DaemonRoutes.LOGS_SOURCE], request.url.parameters[DaemonRoutes.LOGS_FILTER])
            }

            DaemonRoutes.SHUTDOWN -> {
                running = false
                server = ServerDescription()
                discovery.delete()
                json(ShutdownResponse.serializer(), ShutdownResponse("stopping"))
            }

            else -> {
                respond("", HttpStatusCode.NotFound)
            }
        }
    }

    private fun status() =
        DaemonStatus(
            version,
            PID,
            port,
            uptimeSeconds = 125,
            EngineDescription("fake", "0", listOf("cpu")),
            instances.size,
            server,
            logFile = LOG_FILE,
        )

    /** A log that has [LOG] in it and then ends, as the log of a daemon that stops. */
    private fun MockRequestHandleScope.logs(
        source: String?,
        filter: String?,
    ): HttpResponseData {
        val events = LOG.filter { (source == null || it.source == source) && (filter == null || it.type == filter) }
        return respond(
            "event: ${DaemonRoutes.EVENT_LISTENING}\ndata: {}\n\n: keep-alive\n\n" +
                events.joinToString("") { event(DaemonRoutes.EVENT_LOG, LogEvent.serializer(), it) },
            HttpStatusCode.OK,
            headersOf(HttpHeaders.ContentType, "text/event-stream"),
        )
    }

    private fun MockRequestHandleScope.serve(request: ServerRequest): HttpResponseData {
        val asked = ServerDescription(request.running, request.bind, request.port, request.cors, request.apiKey != null)
        if (request.running && server.running && server != asked) {
            return invalid("The server already runs on ${server.bind}:${server.port}; stop it first with `runa server stop`")
        }
        server = if (request.running) asked else ServerDescription()
        return json(ServerDescription.serializer(), server)
    }

    private fun MockRequestHandleScope.load(request: LoadRequest): HttpResponseData {
        val model =
            MODELS.firstOrNull { it.key.contains(request.model, ignoreCase = true) }
                ?: return invalid("No model matches '${request.model}' in /models")
        val context = request.contextLength ?: 4096
        val estimate = MemoryEstimateDescription(model.sizeBytes, context * 1_000L, model.sizeBytes + context * 1_000L)
        if (request.estimateOnly) return json(LoadResponse.serializer(), LoadResponse(LoadResponse.ESTIMATED, estimate))
        val id = request.identifier ?: model.key
        if (instances.any { it.id == id }) return invalid("A model is already loaded as '$id'; unload it or choose another identifier")
        val instance =
            InstanceDescription(
                id,
                model.key,
                context,
                request.gpu ?: "max",
                request.parallel ?: 1,
                request.ttl?.takeIf {
                    it > 0
                },
                idleSeconds = 0,
                model.sizeBytes,
            )
        instances += instance
        val loaded = LoadResponse(LoadResponse.LOADED, estimate, instance, loadTimeSeconds = 1.5)
        check(request.stream) { "the CLI always asks for progress" }
        val events =
            listOf(0f, 0.5f, 1f).joinToString("") { event(DaemonRoutes.EVENT_PROGRESS, LoadProgress.serializer(), LoadProgress(it)) } +
                event(DaemonRoutes.EVENT_LOADED, LoadResponse.serializer(), loaded)
        return respond(events, HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "text/event-stream"))
    }

    /**
     * Answers with what it was asked, and with what the request carried besides: `[system prompt]`
     * in front, and the answer it continues behind. [FAILING] as the input fails halfway.
     */
    private fun MockRequestHandleScope.chat(request: ChatRequest): HttpResponseData {
        check(request.stream) { "the CLI always streams" }
        if (instances.none { it.id == request.model }) {
            val error = ApiError.Detail("No model is called '${request.model}'", ApiError.INVALID_REQUEST, code = ApiError.MODEL_NOT_FOUND)
            return json(ApiError.serializer(), ApiError(error), HttpStatusCode.NotFound)
        }
        val input = (request.input as JsonPrimitive).content
        val opening =
            event(DaemonRoutes.EVENT_CHAT_START, ChatStarted.serializer(), ChatStarted(modelInstanceId = request.model)) +
                event(DaemonRoutes.EVENT_MESSAGE_START, ChatMark.serializer(), ChatMark(DaemonRoutes.EVENT_MESSAGE_START))

        fun delta(text: String) = event(DaemonRoutes.EVENT_MESSAGE_DELTA, ChatDelta.serializer(), ChatDelta(content = text))

        fun stream(events: String) = respond(opening + events, HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "text/event-stream"))
        if (input == FAILING) {
            val error = ApiError.Detail("The model fell over", ApiError.SERVER_ERROR)
            return stream(delta("Half an") + event(DaemonRoutes.EVENT_ERROR, ChatFailed.serializer(), ChatFailed(error = error)))
        }
        val head =
            request.systemPrompt
                ?.takeIf { it.isNotEmpty() }
                ?.let { "[$it] " }
                .orEmpty() + "echo:"
        val tail = " $input" + request.previousResponseId?.let { " (after $it)" }.orEmpty()
        val result =
            ChatResponse(
                modelInstanceId = request.model,
                output = listOf(ChatResponse.Output(ChatResponse.MESSAGE, head + tail)),
                stats =
                    ChatStats(
                        inputTokens = 7,
                        totalOutputTokens = 3,
                        tokensPerSecond = 12.5,
                        timeToFirstTokenSeconds = 0.25,
                        stopReason = "stop",
                    ),
                responseId = if (request.store) "resp_${++answers}" else null,
            )
        return stream(
            delta(head) + delta(tail) +
                event(DaemonRoutes.EVENT_MESSAGE_END, ChatMark.serializer(), ChatMark(DaemonRoutes.EVENT_MESSAGE_END)) +
                event(DaemonRoutes.EVENT_CHAT_END, ChatEnded.serializer(), ChatEnded(result = result)),
        )
    }

    private fun MockRequestHandleScope.unload(request: UnloadRequest): HttpResponseData {
        val gone =
            if (request.all) {
                instances.map { it.id }
            } else {
                val id = request.instanceId
                if (instances.none { it.id == id }) return invalid("No loaded model is called '$id'. `runa ps` lists the loaded models")
                listOfNotNull(id)
            }
        instances.removeAll { it.id in gone }
        return json(UnloadResponse.serializer(), UnloadResponse(gone))
    }

    private fun MockRequestHandleScope.invalid(message: String) =
        json(ApiError.serializer(), ApiError(ApiError.Detail(message, ApiError.INVALID_REQUEST)), HttpStatusCode.BadRequest)

    private fun <T> MockRequestHandleScope.json(
        serializer: SerializationStrategy<T>,
        value: T,
        status: HttpStatusCode = HttpStatusCode.OK,
    ) = respond(DaemonJson.encodeToString(serializer, value), status, headersOf(HttpHeaders.ContentType, "application/json"))

    private fun <T> event(
        name: String,
        serializer: SerializationStrategy<T>,
        value: T,
    ) = "event: $name\ndata: ${DaemonJson.encodeToString(serializer, value)}\n\n"

    companion object {
        const val PID = 4242
        const val PORT = 41234
        const val TOKEN = "secret-token"
        const val ALPHA = "acme/alpha@Q4_K_M"
        const val BETA = "acme/beta@Q8_0"

        /** The message that makes the model fail halfway through its answer. */
        const val FAILING = "fail"

        private fun model(
            key: String,
            size: Long,
        ) = ModelDescription(key, "llm", "Tiny", "llama", key.substringAfter('@'), 494_000_000, size, 32_768, emptyList())

        val MODELS = listOf(model(ALPHA, 398_000_000), model(BETA, 676_000_000))

        /** Not under the test's own directory, which has another name in every run. */
        const val LOG_FILE = "/state/logs/daemon.log"

        private const val TIME = "2026-01-02T03:04:05Z"

        /** One event of every kind. */
        val LOG =
            listOf(
                LogEvent(TIME, LogEvent.SERVER, LogEvent.LOG, "Loaded $ALPHA as '$ALPHA' in 15 ms", level = "info"),
                LogEvent(TIME, LogEvent.RUNTIME, LogEvent.LOG, "llama_context: n_ctx = 4096", level = "debug"),
                LogEvent(
                    TIME,
                    LogEvent.MODEL,
                    LogEvent.INPUT,
                    "<|user|>\nWhat is the weather?\n<|assistant|>\n",
                    model = ALPHA,
                    endpoint = "chat",
                ),
                LogEvent(
                    TIME,
                    LogEvent.MODEL,
                    LogEvent.OUTPUT,
                    "Let me look.",
                    model = ALPHA,
                    endpoint = "chat",
                    toolCalls = listOf(LogEvent.ToolCall("get_weather", "{\"city\":\"Oslo\"}")),
                    stats = ChatStats(7, 3, 12.5, 0.25, "tool_calls"),
                ),
                LogEvent(TIME, LogEvent.SERVER, LogEvent.REQUEST, "POST /api/v1/chat 200 310 ms", level = "debug"),
            )
    }
}
