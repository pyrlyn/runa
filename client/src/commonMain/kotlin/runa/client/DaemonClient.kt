package runa.client

import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.plugins.timeout
import io.ktor.client.request.parameter
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import kotlinx.serialization.SerializationException
import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.core.api.DaemonStatus
import runa.core.api.LoadProgress
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.LogEvent
import runa.core.api.ModelList
import runa.core.api.ServerDescription
import runa.core.api.ServerRequest
import runa.core.api.ShutdownResponse
import runa.core.api.UnloadRequest
import runa.core.api.UnloadResponse
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

/**
 * Talks to a runa daemon over its native API (PLAN.md, section 6).
 *
 * Every call throws [UserInputError] when the daemon says the request is wrong, and [BackendError]
 * when the daemon cannot be reached, rejects the token or fails.
 *
 * @param engine null for the platform's own HTTP engine; tests pass their own.
 * @param baseUrl for example `http://127.0.0.1:41234`.
 * @param token the bearer token of the daemon's discovery file; null sends none.
 */
public class DaemonClient(
    engine: HttpClientEngine? = null,
    private val baseUrl: String,
    token: String?,
) : AutoCloseable {
    private val api =
        object : ApiTransport(engine, baseUrl, token, DaemonJson) {
            override val service = "the runa daemon"
            override val unreadable = "$baseUrl sent an answer that runa cannot read; is it a runa daemon of another version?"
            override val chatCutShort = "The daemon closed the connection before the answer was complete; see its log"

            override fun failure(
                status: HttpStatusCode,
                body: String,
            ): RuntimeException {
                val error =
                    try {
                        DaemonJson.decodeFromString(ApiError.serializer(), body).error
                    } catch (_: SerializationException) {
                        null
                    }
                return when {
                    status == HttpStatusCode.Unauthorized -> {
                        BackendError("The daemon at $baseUrl rejected the token; give the right one with --token or RUNA_TOKEN")
                    }

                    // Whatever the status: a model nobody has is a 404, and still the user's to put right.
                    error?.type == ApiError.INVALID_REQUEST -> {
                        UserInputError(error.message)
                    }

                    status == HttpStatusCode.BadRequest -> {
                        UserInputError("The daemon rejected the request")
                    }

                    else -> {
                        BackendError("The daemon at $baseUrl answered ${status.value}" + error?.let { ": ${it.message}" }.orEmpty())
                    }
                }
            }

            override fun failure(error: ApiError.Detail): RuntimeException =
                if (error.type == ApiError.INVALID_REQUEST) {
                    UserInputError(error.message)
                } else {
                    BackendError("The answer was cut short: ${error.message}")
                }
        }

    /** The one call that needs no token. A daemon that does not answer within [timeout] counts as unreachable. */
    public suspend fun status(timeout: Duration = STATUS_TIMEOUT): DaemonStatus =
        api.call(HttpMethod.Get, DaemonRoutes.STATUS, configure = { timeout { requestTimeoutMillis = timeout.inWholeMilliseconds } }) {
            api.read(it, DaemonStatus.serializer())
        }

    public suspend fun models(): ModelList = api.get(DaemonRoutes.MODELS, ModelList.serializer())

    /**
     * Loads a model, or only estimates it when the request says so. Returns when the model is
     * ready; [onProgress] gets values from 0 to 1 on the way. Cancelling the call cancels the load.
     */
    public suspend fun load(
        request: LoadRequest,
        onProgress: (Float) -> Unit = {},
    ): LoadResponse {
        if (request.estimateOnly) return api.post(DaemonRoutes.LOAD, LoadRequest.serializer(), request, LoadResponse.serializer())
        val streamed = api.encode(LoadRequest.serializer(), request.copy(stream = true))
        return api.call(HttpMethod.Post, DaemonRoutes.LOAD, streamed) { response ->
            var loaded: LoadResponse? = null
            api.events(response) { event, data ->
                when (event) {
                    DaemonRoutes.EVENT_PROGRESS -> onProgress(api.decode(LoadProgress.serializer(), data).progress)
                    DaemonRoutes.EVENT_LOADED -> loaded = api.decode(LoadResponse.serializer(), data)
                    DaemonRoutes.EVENT_ERROR -> throw api.failure(HttpStatusCode.BadRequest, data)
                }
            }
            loaded ?: throw BackendError("The daemon closed the connection before the model was loaded; see its log")
        }
    }

    /**
     * Sends one turn of a conversation and returns the whole answer; [onText] gets its pieces as
     * the model writes them. Cancelling the call ends the generation.
     */
    public suspend fun chat(
        request: ChatRequest,
        onText: (String) -> Unit = {},
    ): ChatResponse = api.chat(request, onText)

    public suspend fun unload(request: UnloadRequest): UnloadResponse =
        api.post(DaemonRoutes.UNLOAD, UnloadRequest.serializer(), request, UnloadResponse.serializer())

    /** Opens or closes the listener for other programs, and returns what it is now. */
    public suspend fun server(request: ServerRequest): ServerDescription =
        api.post(DaemonRoutes.SERVER, ServerRequest.serializer(), request, ServerDescription.serializer())

    /**
     * Reads the daemon's log as it is written, until the daemon ends the stream, which it does when
     * it stops. [source] is one of [LogEvent.SOURCES] and [filter] one of [LogEvent.FILTERS]; null
     * for everything. [onListening] is called once: no event after it is missed.
     *
     * @throws BackendError when the stream cannot be opened. A stream that breaks later has ended.
     */
    public suspend fun logs(
        source: String? = null,
        filter: String? = null,
        onListening: () -> Unit = {},
        onEvent: (LogEvent) -> Unit,
    ) {
        var listening = false
        try {
            api.call(
                HttpMethod.Get,
                DaemonRoutes.LOGS,
                configure = {
                    source?.let { parameter(DaemonRoutes.LOGS_SOURCE, it) }
                    filter?.let { parameter(DaemonRoutes.LOGS_FILTER, it) }
                },
            ) { response ->
                api.events(response) { event, data ->
                    when (event) {
                        DaemonRoutes.EVENT_LISTENING -> {
                            listening = true
                            onListening()
                        }

                        DaemonRoutes.EVENT_LOG -> {
                            onEvent(api.decode(LogEvent.serializer(), data))
                        }
                    }
                }
            }
        } catch (e: BackendError) {
            // A daemon that stops cuts the connection: that is how this stream ends, not a failure of it.
            if (!listening) throw e
        }
    }

    /** Asks the daemon to stop; it answers first and exits afterwards. */
    public suspend fun shutdown() {
        api.call(HttpMethod.Post, DaemonRoutes.SHUTDOWN) { api.read(it, ShutdownResponse.serializer()) }
    }

    override fun close(): Unit = api.close()

    private companion object {
        val STATUS_TIMEOUT = 5.seconds
    }
}
