package runa.core.api

import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * One event of the daemon's log, as `GET /api/v1/logs/stream` sends it and `runa log stream --json`
 * prints it. Which fields are set depends on [type]; a field that is not set is left out, so that
 * a line of the log stays short.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
public data class LogEvent(
    /** When it happened, as an ISO 8601 instant. */
    val time: String,
    /** [MODEL], [SERVER] or [RUNTIME]. */
    val source: String,
    /** [INPUT] and [OUTPUT] of a model, a [REQUEST] the server answered, or a [LOG] line. */
    val type: String,
    /** The prompt as the model read it, what the model wrote, or the line. */
    val text: String,
    /** `debug`, `info`, `warn` or `error`; only a [LOG] line and a [REQUEST] have one. */
    @EncodeDefault(EncodeDefault.Mode.NEVER) val level: String? = null,
    /** The loaded instance that was asked; only for [INPUT] and [OUTPUT]. */
    @EncodeDefault(EncodeDefault.Mode.NEVER) val model: String? = null,
    /** Which endpoint asked it: `chat`, `chat.completions` or `completions`; only for [INPUT] and [OUTPUT]. */
    @EncodeDefault(EncodeDefault.Mode.NEVER) val endpoint: String? = null,
    /** The tools an [OUTPUT] called. */
    @EncodeDefault(EncodeDefault.Mode.NEVER) @SerialName("tool_calls") val toolCalls: List<ToolCall>? = null,
    /** How the generation of an [OUTPUT] went. */
    @EncodeDefault(EncodeDefault.Mode.NEVER) val stats: ChatStats? = null,
) {
    @Serializable
    public data class ToolCall(
        val name: String,
        /** The arguments as the model wrote them: JSON, when the model got it right. */
        val arguments: String,
    )

    public companion object {
        /** What the models read and wrote. */
        public const val MODEL: String = "model"

        /** The daemon's own log and the requests it answered. */
        public const val SERVER: String = "server"

        /** The log of the inference engine. */
        public const val RUNTIME: String = "runtime"

        public val SOURCES: List<String> = listOf(MODEL, SERVER, RUNTIME)

        public const val INPUT: String = "input"
        public const val OUTPUT: String = "output"
        public const val REQUEST: String = "request"
        public const val LOG: String = "log"

        /** What a stream of the [MODEL] source can be narrowed to. */
        public val FILTERS: List<String> = listOf(INPUT, OUTPUT)
    }
}
