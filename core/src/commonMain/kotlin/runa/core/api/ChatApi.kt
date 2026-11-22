package runa.core.api

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

// `POST /api/v1/chat`, in the shape of LM Studio's endpoint of the same name, so a client written
// for one can talk to the other. What runa does not have yet (images, reasoning, tool calls) is
// left out rather than faked.

/**
 * One turn of a conversation. The daemon keeps the turns it answered: a request continues the
 * conversation that ended with [previousResponseId], and sends only what the user said next.
 */
@Serializable
public data class ChatRequest(
    /** An identifier from `runa ps` or a key from `runa ls`; a model that is not loaded is loaded first. */
    val model: String,
    /** What the user says: a string, or a list of `{"type": "text", "content": "…"}` parts. */
    val input: JsonElement,
    /** Null keeps the system prompt of the conversation that is continued; an empty one removes it. */
    @SerialName("system_prompt") val systemPrompt: String? = null,
    /** True for server-sent events (see [DaemonRoutes]) instead of one [ChatResponse]. */
    val stream: Boolean = false,
    val temperature: Double? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    @SerialName("min_p") val minP: Double? = null,
    @SerialName("repeat_penalty") val repeatPenalty: Double? = null,
    @SerialName("max_output_tokens") val maxOutputTokens: Int? = null,
    /** False answers without keeping the turn, so nothing can continue it. */
    val store: Boolean = true,
    @SerialName("previous_response_id") val previousResponseId: String? = null,
    /** Tools for the model. Not supported: a request with any is refused rather than answered without them. */
    val integrations: JsonElement? = null,
)

/** The answer to a [ChatRequest], and the `result` of the last event of a streamed one. */
@Serializable
public data class ChatResponse(
    /** The loaded instance that answered. */
    @SerialName("model_instance_id") val modelInstanceId: String,
    val output: List<Output>,
    val stats: ChatStats,
    /** What the next request gives as `previous_response_id`; null when the turn was not stored. */
    @SerialName("response_id") val responseId: String? = null,
) {
    @Serializable
    public data class Output(
        /** Always [MESSAGE] today. */
        val type: String,
        val content: String,
    )

    /** What the model said. */
    public val text: String
        get() = output.filter { it.type == MESSAGE }.joinToString("") { it.content }

    public companion object {
        public const val MESSAGE: String = "message"
    }
}

@Serializable
public data class ChatStats(
    /** The whole prompt: every earlier turn, the system prompt and the template around them. */
    @SerialName("input_tokens") val inputTokens: Int,
    @SerialName("total_output_tokens") val totalOutputTokens: Int,
    @SerialName("tokens_per_second") val tokensPerSecond: Double,
    @SerialName("time_to_first_token_seconds") val timeToFirstTokenSeconds: Double,
    /** `stop`: the model ended its answer; `length`: `max_output_tokens` or the context ran out. Not in LM Studio's stats. */
    @SerialName("stop_reason") val stopReason: String,
)

// The data of the events of a streamed chat. Each says its own event name as `type`, as LM Studio's do.

/** `chat.start`, the first event. */
@Serializable
public data class ChatStarted(
    val type: String = DaemonRoutes.EVENT_CHAT_START,
    @SerialName("model_instance_id") val modelInstanceId: String,
)

/** `message.start` and `message.end`, around the text of the answer. */
@Serializable
public data class ChatMark(
    val type: String,
)

/** `message.delta`: the next piece of the answer. */
@Serializable
public data class ChatDelta(
    val type: String = DaemonRoutes.EVENT_MESSAGE_DELTA,
    val content: String,
)

/** `error`: the last event of a chat that failed after its stream had begun. */
@Serializable
public data class ChatFailed(
    val type: String = DaemonRoutes.EVENT_ERROR,
    val error: ApiError.Detail,
)

/** `chat.end`, the last event of a chat that was answered. */
@Serializable
public data class ChatEnded(
    val type: String = DaemonRoutes.EVENT_CHAT_END,
    val result: ChatResponse,
)
