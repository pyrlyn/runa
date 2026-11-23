package runa.daemon

import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.header
import io.ktor.server.response.respondBytesWriter
import io.ktor.utils.io.writeStringUtf8
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.produceIn
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.engine.GenerationEvent
import runa.engine.GenerationStats
import runa.engine.Sampling
import kotlin.math.roundToInt

// What the endpoints that generate text have in common, whichever API they belong to.

/** One server-sent event, ready to send. */
internal fun sseFrame(
    data: String,
    event: String? = null,
): String = event?.let { "event: $it\n" }.orEmpty() + "data: $data\n\n"

/** How one endpoint writes a generation as server-sent events. Each function returns whole frames, see [sseFrame]. */
internal interface EventFormat {
    /** What is sent before any text. */
    fun opening(): List<String>

    fun text(text: String): List<String>

    /** A tool call begins. Only an endpoint that gives the model tools is asked. */
    fun toolCall(call: GenerationEvent.ToolCallStarted): List<String> = error("This endpoint gives the model no tools")

    /** The next piece of a call's arguments. */
    fun toolArguments(arguments: GenerationEvent.ToolCallArguments): List<String> = error("This endpoint gives the model no tools")

    /** The frames that end a generation. This is also where an endpoint learns that the answer is complete. */
    suspend fun finished(event: GenerationEvent.Finished): List<String>

    /** The last frames of a generation that failed after its stream had begun. */
    fun failed(error: ApiError): List<String>
}

/**
 * Answers with [events] as a stream in [format]. A generation that fails before it produced
 * anything throws, so the caller can still answer with a status code; a later failure is the last
 * event of the stream.
 */
internal suspend fun ApplicationCall.respondEvents(
    events: Flow<GenerationEvent>,
    format: EventFormat,
    failures: Failures,
) = coroutineScope {
    // A failure travels as a value: thrown, it would cancel this scope and with it the answer.
    val results =
        events
            .map { Result.success(it) }
            .catch { emit(Result.failure(it)) }
            .produceIn(this)
    // Waited for before the answer begins: a prompt the model cannot take is known by now.
    val first = results.receive().getOrThrow()
    response.header(HttpHeaders.CacheControl, "no-cache")
    // A client that goes away fails the next write, which ends this scope and the generation.
    respondBytesWriter(ContentType.Text.EventStream) {
        suspend fun send(frames: List<String>) {
            for (frame in frames) {
                writeStringUtf8(frame)
                flush()
            }
        }

        suspend fun send(event: GenerationEvent) =
            send(
                when (event) {
                    is GenerationEvent.Text -> format.text(event.text)
                    is GenerationEvent.ToolCallStarted -> format.toolCall(event)
                    is GenerationEvent.ToolCallArguments -> format.toolArguments(event)
                    is GenerationEvent.Finished -> format.finished(event)
                },
            )
        send(format.opening())
        send(first)
        for (result in results) {
            val event = result.getOrNull()
            if (event == null) {
                // The status line is long gone, so the failure is the last thing the client reads.
                send(format.failed(failures.describe(checkNotNull(result.exceptionOrNull())).second))
                return@respondBytesWriter
            }
            send(event)
        }
    }
}

/** The engine's defaults, with what a request gave instead. */
internal fun samplingOf(
    temperature: Double?,
    topP: Double?,
    topK: Int?,
    minP: Double?,
    repeatPenalty: Double?,
    seed: Long? = null,
): Sampling {
    if (temperature != null && temperature < 0) throw UserInputError("temperature must not be negative")
    return Sampling().let {
        it.copy(
            temperature = temperature ?: it.temperature,
            topP = topP ?: it.topP,
            topK = topK ?: it.topK,
            minP = minP ?: it.minP,
            repeatPenalty = repeatPenalty ?: it.repeatPenalty,
            seed = seed,
        )
    }
}

/** The most tokens to generate. A negative limit is how llama.cpp and LM Studio clients ask for none. */
internal fun tokenLimit(
    limit: Int?,
    field: String,
): Int? {
    if (limit == 0) throw UserInputError("$field must be at least 1")
    return limit?.takeIf { it > 0 }
}

internal fun JsonElement?.hasItems(): Boolean = this is JsonArray && isNotEmpty()

/** One line in the daemon's log for every generation that finished. */
internal fun DaemonLog.report(
    endpoint: String,
    model: String,
    stats: GenerationStats,
) {
    val speed = (stats.tokensPerSecond * 10).roundToInt() / 10.0
    info("$endpoint '$model': ${stats.promptTokens} prompt tokens, ${stats.completionTokens} generated, $speed tokens/s")
}
