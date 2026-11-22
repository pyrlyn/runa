package runa.engine

import kotlinx.coroutines.flow.Flow
import runa.core.ChatMessage
import kotlin.time.Duration
import kotlin.time.DurationUnit

public data class GenerationRequest(
    val prompt: Prompt,
    val sampling: Sampling = Sampling(),
    /** The most tokens to generate; null means until the model stops or the context is full. */
    val maxTokens: Int? = null,
    /** Generation ends before any of these strings would be emitted. */
    val stop: List<String> = emptyList(),
)

public sealed interface Prompt {
    /** Sent to the model as is. */
    public data class Text(
        val text: String,
    ) : Prompt

    /** Rendered with the model's own chat template. */
    public data class Chat(
        val messages: List<ChatMessage>,
    ) : Prompt
}

/** Sampler settings; the defaults are llama.cpp's. */
public data class Sampling(
    val temperature: Double = 0.8,
    val topP: Double = 0.95,
    val topK: Int = 40,
    val minP: Double = 0.05,
    val repeatPenalty: Double = 1.0,
    /** Null means a random seed; the same seed with temperature 0 gives the same output. */
    val seed: Long? = null,
)

public sealed interface GenerationEvent {
    /** A piece of generated text: one token, or several when a token ends inside a UTF-8 sequence. */
    public data class Text(
        val text: String,
    ) : GenerationEvent

    public data class Finished(
        val reason: FinishReason,
        val stats: GenerationStats,
    ) : GenerationEvent
}

public enum class FinishReason {
    /** The model ended its turn or produced a stop string. */
    STOP,

    /** [GenerationRequest.maxTokens] or the context length was reached. */
    LENGTH,
}

public data class GenerationStats(
    val promptTokens: Int,
    val completionTokens: Int,
    val timeToFirstToken: Duration,
    /** Generation time after the first token; with [completionTokens] it gives the speed. */
    val generationTime: Duration,
) {
    val tokensPerSecond: Double
        get() {
            val seconds = generationTime.toDouble(DurationUnit.SECONDS)
            return if (seconds > 0) completionTokens / seconds else 0.0
        }
}

/** A whole completion, for callers that do not stream. */
public data class Completion(
    val text: String,
    val reason: FinishReason,
    val stats: GenerationStats,
)

/** Collects a [LoadedModel.generate] flow into one [Completion]. */
public suspend fun Flow<GenerationEvent>.toCompletion(): Completion {
    val text = StringBuilder()
    var finished: GenerationEvent.Finished? = null
    collect { event ->
        when (event) {
            is GenerationEvent.Text -> text.append(event.text)
            is GenerationEvent.Finished -> finished = event
        }
    }
    val end = checkNotNull(finished) { "The engine ended generation without a Finished event" }
    return Completion(text.toString(), end.reason, end.stats)
}
