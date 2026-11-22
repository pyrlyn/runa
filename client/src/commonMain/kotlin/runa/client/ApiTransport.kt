package runa.client

import io.ktor.client.HttpClient
import io.ktor.client.HttpClientConfig
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.plugins.HttpTimeout
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
import kotlinx.serialization.json.Json
import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.ChatDelta
import runa.core.api.ChatEnded
import runa.core.api.ChatFailed
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.DaemonRoutes
import kotlin.time.Duration.Companion.seconds

/**
 * What the clients of both backends share: one HTTP request with its bearer token, a failed one as
 * a runa error, JSON bodies, server-sent events, and the chat, which both services speak alike.
 * A subclass says how its service names itself and its failures.
 *
 * @param engine null for the platform's own HTTP engine; tests pass their own.
 */
internal abstract class ApiTransport(
    engine: HttpClientEngine?,
    protected val baseUrl: String,
    private val token: String?,
    private val json: Json,
) : AutoCloseable {
    private val http = if (engine != null) HttpClient(engine) { configure() } else HttpClient { configure() }

    private fun HttpClientConfig<*>.configure() {
        install(HttpTimeout) { connectTimeoutMillis = CONNECT_TIMEOUT.inWholeMilliseconds }
    }

    /** For messages: `the runa daemon`. */
    protected abstract val service: String

    /** What to say about an answer that is not the JSON this build expects. */
    protected abstract val unreadable: String

    /** What to say when the stream of a chat ends without its last event. */
    protected abstract val chatCutShort: String

    /** What a response that is not a success means. */
    abstract fun failure(
        status: HttpStatusCode,
        body: String,
    ): RuntimeException

    /** What the `error` event of a stream means: the answer began, so there was no status to read it from. */
    abstract fun failure(error: ApiError.Detail): RuntimeException

    suspend fun <T> call(
        method: HttpMethod,
        path: String,
        body: String? = null,
        configure: HttpRequestBuilder.() -> Unit = {},
        read: suspend (HttpResponse) -> T,
    ): T =
        try {
            http
                .prepareRequest(baseUrl + path) {
                    this.method = method
                    token?.let { header(HttpHeaders.Authorization, "Bearer $it") }
                    body?.let {
                        contentType(ContentType.Application.Json)
                        setBody(it)
                    }
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
            throw BackendError("Cannot reach $service at $baseUrl: ${e.message ?: e::class.simpleName}", e)
        }

    suspend fun <T> get(
        path: String,
        answer: DeserializationStrategy<T>,
    ): T = call(HttpMethod.Get, path) { read(it, answer) }

    suspend fun <B, T> post(
        path: String,
        serializer: SerializationStrategy<B>,
        body: B,
        answer: DeserializationStrategy<T>,
    ): T = call(HttpMethod.Post, path, encode(serializer, body)) { read(it, answer) }

    fun <T> encode(
        serializer: SerializationStrategy<T>,
        value: T,
    ): String = json.encodeToString(serializer, value)

    suspend fun <T> read(
        response: HttpResponse,
        serializer: DeserializationStrategy<T>,
    ): T = decode(serializer, response.bodyAsText())

    fun <T> decode(
        serializer: DeserializationStrategy<T>,
        text: String,
    ): T =
        try {
            json.decodeFromString(serializer, text)
        } catch (e: SerializationException) {
            throw BackendError(unreadable, e)
        }

    /** Server-sent events: `event:` and `data:` lines, ended by an empty line. */
    suspend fun events(
        response: HttpResponse,
        handle: (event: String, data: String) -> Unit,
    ) {
        val channel = response.bodyAsChannel()
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

    /**
     * Sends one turn of a conversation and returns the whole answer; [onText] gets its pieces as
     * the model writes them. Cancelling the call ends the generation.
     */
    suspend fun chat(
        request: ChatRequest,
        onText: (String) -> Unit,
    ): ChatResponse =
        call(HttpMethod.Post, DaemonRoutes.CHAT, encode(ChatRequest.serializer(), request.copy(stream = true))) { response ->
            var result: ChatResponse? = null
            events(response) { event, data ->
                when (event) {
                    DaemonRoutes.EVENT_MESSAGE_DELTA -> onText(decode(ChatDelta.serializer(), data).content)
                    DaemonRoutes.EVENT_CHAT_END -> result = decode(ChatEnded.serializer(), data).result
                    DaemonRoutes.EVENT_ERROR -> throw failure(decode(ChatFailed.serializer(), data).error)
                }
            }
            result ?: throw BackendError(chatCutShort)
        }

    override fun close(): Unit = http.close()

    private companion object {
        val CONNECT_TIMEOUT = 5.seconds
        const val EVENT_FIELD = "event:"
        const val DATA_FIELD = "data:"
    }
}
