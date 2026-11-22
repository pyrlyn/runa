package runa.daemon

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement

// The JSON of the OpenAI-compatible API (https://platform.openai.com/docs/api-reference). Only the
// daemon writes it: the clients are other people's programs.

/**
 * Requests are read leniently: unknown fields are skipped and an explicit `null` means the default,
 * because clients send both. In an answer a field with a default is left out when it has that
 * value, and a nullable field without one is always written: `finish_reason` is `null` in every
 * chunk but the last, as OpenAI sends it.
 */
internal val OpenAiJson: Json =
    Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
    }

internal object OpenAiRoutes {
    const val MODELS = "/v1/models"
    const val CHAT_COMPLETIONS = "/v1/chat/completions"
    const val COMPLETIONS = "/v1/completions"
    const val EMBEDDINGS = "/v1/embeddings"
}

/** `GET /v1/models`. */
@Serializable
internal data class ModelCardList(
    val `object`: String,
    val data: List<ModelCard>,
)

@Serializable
internal data class ModelCard(
    val id: String,
    val `object`: String,
    val created: Long,
    @SerialName("owned_by") val ownedBy: String,
)

@Serializable
internal data class StreamOptions(
    @SerialName("include_usage") val includeUsage: Boolean = false,
)

/** `POST /v1/chat/completions`. `top_k`, `min_p` and `repeat_penalty` are llama.cpp's additions. */
@Serializable
internal data class ChatCompletionRequest(
    val model: String,
    val messages: List<Message>,
    val stream: Boolean = false,
    @SerialName("stream_options") val streamOptions: StreamOptions? = null,
    @SerialName("max_tokens") val maxTokens: Int? = null,
    @SerialName("max_completion_tokens") val maxCompletionTokens: Int? = null,
    val temperature: Double? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    @SerialName("min_p") val minP: Double? = null,
    @SerialName("repeat_penalty") val repeatPenalty: Double? = null,
    val seed: Long? = null,
    /** A string or a list of strings. */
    val stop: JsonElement? = null,
    val n: Int? = null,
    val tools: JsonElement? = null,
    val functions: JsonElement? = null,
    @SerialName("response_format") val responseFormat: JsonElement? = null,
) {
    @Serializable
    data class Message(
        val role: String,
        /** A string, or a list of `{"type": "text", "text": ...}` parts. */
        val content: JsonElement? = null,
    )
}

@Serializable
internal data class Usage(
    @SerialName("prompt_tokens") val promptTokens: Int,
    @SerialName("completion_tokens") val completionTokens: Int,
    @SerialName("total_tokens") val totalTokens: Int,
)

@Serializable
internal data class ChatCompletion(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<Choice>,
    val usage: Usage,
) {
    @Serializable
    data class Choice(
        val index: Int,
        val message: Message,
        @SerialName("finish_reason") val finishReason: String,
    )

    @Serializable
    data class Message(
        val role: String,
        val content: String,
    )
}

/** One `data:` event of a streamed chat completion. */
@Serializable
internal data class ChatCompletionChunk(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<Choice>,
    /** Only in the extra last chunk that `stream_options.include_usage` asks for. */
    val usage: Usage? = null,
) {
    @Serializable
    data class Choice(
        val index: Int,
        val delta: Delta,
        @SerialName("finish_reason") val finishReason: String?,
    )

    @Serializable
    data class Delta(
        val role: String? = null,
        val content: String? = null,
    )
}

/** `POST /v1/completions`. */
@Serializable
internal data class CompletionRequest(
    val model: String,
    /** A string, or a list with one string. */
    val prompt: JsonElement,
    val stream: Boolean = false,
    @SerialName("stream_options") val streamOptions: StreamOptions? = null,
    @SerialName("max_tokens") val maxTokens: Int? = null,
    val temperature: Double? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    @SerialName("min_p") val minP: Double? = null,
    @SerialName("repeat_penalty") val repeatPenalty: Double? = null,
    val seed: Long? = null,
    val stop: JsonElement? = null,
    val n: Int? = null,
    val echo: Boolean = false,
    val suffix: String? = null,
)

/** The answer of `/v1/completions`, and each event of its stream. */
@Serializable
internal data class TextCompletion(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<Choice>,
    val usage: Usage? = null,
) {
    @Serializable
    data class Choice(
        val index: Int,
        val text: String,
        @SerialName("finish_reason") val finishReason: String?,
    )
}

/** `POST /v1/embeddings`. */
@Serializable
internal data class EmbeddingRequest(
    val model: String,
    /** A string or a list of strings. */
    val input: JsonElement,
    @SerialName("encoding_format") val encodingFormat: String = FLOAT,
    val dimensions: Int? = null,
) {
    companion object {
        const val FLOAT = "float"
        const val BASE64 = "base64"
    }
}

@Serializable
internal data class EmbeddingList(
    val `object`: String,
    val data: List<Embedding>,
    val model: String,
    val usage: EmbeddingUsage,
)

@Serializable
internal data class Embedding(
    val `object`: String,
    val index: Int,
    /** A list of numbers, or for `base64` the little-endian 32-bit floats as one string. */
    val embedding: JsonElement,
)

@Serializable
internal data class EmbeddingUsage(
    @SerialName("prompt_tokens") val promptTokens: Int,
    @SerialName("total_tokens") val totalTokens: Int,
)
