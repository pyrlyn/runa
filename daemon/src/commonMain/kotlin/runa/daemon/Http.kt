package runa.daemon

import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.receiveText
import io.ktor.server.response.respondText
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.DeserializationStrategy
import kotlinx.serialization.SerializationException
import kotlinx.serialization.SerializationStrategy
import kotlinx.serialization.json.Json
import runa.core.LogLevel
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.DaemonJson
import kotlin.concurrent.Volatile

/** Which of the daemon's two listeners a request arrived on. */
internal enum class Listener {
    /** The loopback port of the discovery file, which the CLI talks to. */
    PRIVATE,

    /** What `runa server start` opened for other programs. */
    PUBLIC,
}

/**
 * Who may call what. The daemon's own [token] opens everything. The OpenAI-compatible routes are
 * what other programs use, and those have no way to read the discovery file: on the public
 * listener they are open, or take the API key that `server start` was given.
 */
internal class Access(
    val token: String,
) {
    /** Null while the public listener asks for no key. */
    @Volatile
    var apiKey: String? = null

    /** For the native API on either listener. */
    fun requireToken(call: ApplicationCall) {
        if (!matches(bearer(call), token)) throw Unauthorized()
    }

    /** For the OpenAI-compatible routes. */
    fun requireInference(
        call: ApplicationCall,
        listener: Listener,
    ) {
        val given = bearer(call)
        if (matches(given, token)) return
        val key = apiKey
        if (listener == Listener.PRIVATE || (key != null && !matches(given, key))) throw Unauthorized()
    }

    private fun bearer(call: ApplicationCall): String? =
        call.request.headers[HttpHeaders.Authorization]
            ?.takeIf { it.startsWith(BEARER, ignoreCase = true) }
            ?.substring(BEARER.length)

    private fun matches(
        given: String?,
        secret: String,
    ): Boolean {
        if (given == null) return false
        val a = given.encodeToByteArray()
        val b = secret.encodeToByteArray()
        // Every byte is compared, so the time taken says nothing about where the two differ.
        var difference = a.size xor b.size
        for (i in a.indices) difference = difference or (a[i].toInt() xor b[i % b.size].toInt())
        return difference == 0
    }

    private companion object {
        const val BEARER = "Bearer "
    }
}

internal class Unauthorized : RuntimeException()

/** Turns what a route throws into the API's error body, the same for both APIs. */
internal class Failures(
    private val log: DaemonLog,
) {
    suspend fun answer(
        call: ApplicationCall,
        block: suspend () -> Unit,
    ) {
        try {
            block()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            // Once a stream has begun there is no status left to set; this is a client that went away.
            if (call.response.isCommitted) {
                log.at(LogLevel.DEBUG, "${call.request.local.uri} ended early: ${e.message ?: e::class.simpleName}")
                return
            }
            val (status, body) = describe(e)
            call.respondJson(DaemonJson, ApiError.serializer(), body, status)
        }
    }

    fun describe(error: Throwable): Pair<HttpStatusCode, ApiError> {
        fun body(
            message: String,
            type: String,
            code: String? = null,
        ) = ApiError(ApiError.Detail(message, type, code = code))
        return when (error) {
            is Unauthorized -> {
                HttpStatusCode.Unauthorized to
                    body("A valid bearer token is required", ApiError.UNAUTHORIZED, ApiError.INVALID_API_KEY)
            }

            is UnknownModelError -> {
                HttpStatusCode.NotFound to body(error.message.orEmpty(), ApiError.INVALID_REQUEST, ApiError.MODEL_NOT_FOUND)
            }

            is UserInputError -> {
                HttpStatusCode.BadRequest to body(error.message.orEmpty(), ApiError.INVALID_REQUEST)
            }

            is SerializationException -> {
                HttpStatusCode.BadRequest to body("The request body is not valid: ${error.message}", ApiError.INVALID_REQUEST)
            }

            else -> {
                log.error("Request failed: ${error.stackTraceToString()}")
                HttpStatusCode.InternalServerError to body(error.message ?: error.toString(), ApiError.SERVER_ERROR)
            }
        }
    }
}

internal suspend fun <T> ApplicationCall.respondJson(
    json: Json,
    serializer: SerializationStrategy<T>,
    value: T,
    status: HttpStatusCode = HttpStatusCode.OK,
) = respondText(json.encodeToString(serializer, value), ContentType.Application.Json, status)

internal suspend fun <T> ApplicationCall.receiveJson(
    json: Json,
    serializer: DeserializationStrategy<T>,
): T = json.decodeFromString(serializer, receiveText().ifBlank { "{}" })
