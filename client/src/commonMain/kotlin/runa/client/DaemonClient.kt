package runa.client

import io.ktor.client.HttpClient
import io.ktor.client.HttpClientConfig
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.timeout
import io.ktor.client.request.HttpRequestBuilder
import io.ktor.client.request.header
import io.ktor.client.request.prepareRequest
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsChannel
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.utils.io.readLine
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.DeserializationStrategy
import kotlinx.serialization.SerializationException
import kotlinx.serialization.SerializationStrategy
import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.core.api.DaemonStatus
import runa.core.api.LoadProgress
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
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
    private val token: String?,
) : AutoCloseable {
    private val http = if (engine != null) HttpClient(engine) { configure() } else HttpClient { configure() }

    private fun HttpClientConfig<*>.configure() {
        install(HttpTimeout) { connectTimeoutMillis = CONNECT_TIMEOUT.inWholeMilliseconds }
    }

    /** The one call that needs no token. A daemon that does not answer within [timeout] counts as unreachable. */
    public suspend fun status(timeout: Duration = STATUS_TIMEOUT): DaemonStatus =
        call(HttpMethod.Get, DaemonRoutes.STATUS, { timeout { requestTimeoutMillis = timeout.inWholeMilliseconds } }) {
            it.read(DaemonStatus.serializer())
        }

    public suspend fun models(): ModelList = call(HttpMethod.Get, DaemonRoutes.MODELS) { it.read(ModelList.serializer()) }

    /**
     * Loads a model, or only estimates it when the request says so. Returns when the model is
     * ready; [onProgress] gets values from 0 to 1 on the way. Cancelling the call cancels the load.
     */
    public suspend fun load(
        request: LoadRequest,
        onProgress: (Float) -> Unit = {},
    ): LoadResponse {
        if (request.estimateOnly) {
            return call(
                HttpMethod.Post,
                DaemonRoutes.LOAD,
                { json(LoadRequest.serializer(), request) },
            ) { it.read(LoadResponse.serializer()) }
        }
        return call(HttpMethod.Post, DaemonRoutes.LOAD, { json(LoadRequest.serializer(), request.copy(stream = true)) }) { response ->
            var loaded: LoadResponse? = null
            response.events { event, data ->
                when (event) {
                    DaemonRoutes.EVENT_PROGRESS -> onProgress(decode(LoadProgress.serializer(), data).progress)
                    DaemonRoutes.EVENT_LOADED -> loaded = decode(LoadResponse.serializer(), data)
                    DaemonRoutes.EVENT_ERROR -> throw failure(HttpStatusCode.BadRequest, data)
                }
            }
            loaded ?: throw BackendError("The daemon closed the connection before the model was loaded; see its log")
        }
    }

    public suspend fun unload(request: UnloadRequest): UnloadResponse =
        call(HttpMethod.Post, DaemonRoutes.UNLOAD, { json(UnloadRequest.serializer(), request) }) { it.read(UnloadResponse.serializer()) }

    /** Opens or closes the listener for other programs, and returns what it is now. */
    public suspend fun server(request: ServerRequest): ServerDescription =
        call(
            HttpMethod.Post,
            DaemonRoutes.SERVER,
            { json(ServerRequest.serializer(), request) },
        ) { it.read(ServerDescription.serializer()) }

    /** Asks the daemon to stop; it answers first and exits afterwards. */
    public suspend fun shutdown() {
        call(HttpMethod.Post, DaemonRoutes.SHUTDOWN) { it.read(ShutdownResponse.serializer()) }
    }

    override fun close(): Unit = http.close()

    private suspend fun <T> call(
        method: HttpMethod,
        path: String,
        configure: HttpRequestBuilder.() -> Unit = {},
        read: suspend (HttpResponse) -> T,
    ): T =
        try {
            http
                .prepareRequest(baseUrl + path) {
                    this.method = method
                    token?.let { header(HttpHeaders.Authorization, "Bearer $it") }
                    configure()
                }.execute { response ->
                    if (!response.status.isSuccess()) throw failure(response.status, response.bodyAsText())
                    read(response)
                }
        } catch (e: CancellationException) {
            throw e
        } catch (e: UserInputError) {
            throw e
        } catch (e: BackendError) {
            throw e
        } catch (e: Exception) {
            throw BackendError("Cannot reach the runa daemon at $baseUrl: ${e.message ?: e::class.simpleName}", e)
        }

    private fun <T> HttpRequestBuilder.json(
        serializer: SerializationStrategy<T>,
        value: T,
    ) {
        contentType(ContentType.Application.Json)
        setBody(DaemonJson.encodeToString(serializer, value))
    }

    private suspend fun <T> HttpResponse.read(serializer: DeserializationStrategy<T>): T = decode(serializer, bodyAsText())

    private fun <T> decode(
        serializer: DeserializationStrategy<T>,
        text: String,
    ): T =
        try {
            DaemonJson.decodeFromString(serializer, text)
        } catch (e: SerializationException) {
            throw BackendError("$baseUrl sent an answer that runa cannot read; is it a runa daemon of another version?", e)
        }

    /** Server-sent events: `event:` and `data:` lines, ended by an empty line. */
    private suspend fun HttpResponse.events(handle: (event: String, data: String) -> Unit) {
        val channel = bodyAsChannel()
        var event = ""
        val data = StringBuilder()
        while (true) {
            val line = channel.readLine() ?: break
            when {
                line.isEmpty() -> {
                    if (event.isNotEmpty() || data.isNotEmpty()) handle(event, data.toString())
                    event = ""
                    data.clear()
                }

                line.startsWith(EVENT_FIELD) -> {
                    event = line.substring(EVENT_FIELD.length).trim()
                }

                line.startsWith(DATA_FIELD) -> {
                    data.append(line.substring(DATA_FIELD.length).removePrefix(" "))
                }
            }
        }
    }

    private fun failure(
        status: HttpStatusCode,
        body: String,
    ): RuntimeException {
        val message =
            try {
                DaemonJson.decodeFromString(ApiError.serializer(), body).error.message
            } catch (_: SerializationException) {
                null
            }
        return when (status) {
            HttpStatusCode.BadRequest -> {
                UserInputError(message ?: "The daemon rejected the request")
            }

            HttpStatusCode.Unauthorized -> {
                BackendError(
                    "The daemon at $baseUrl rejected the token; give the right one with --token or RUNA_TOKEN",
                )
            }

            else -> {
                BackendError("The daemon at $baseUrl answered ${status.value}" + message?.let { ": $it" }.orEmpty())
            }
        }
    }

    private companion object {
        val CONNECT_TIMEOUT = 5.seconds
        val STATUS_TIMEOUT = 5.seconds
        const val EVENT_FIELD = "event:"
        const val DATA_FIELD = "data:"
    }
}
