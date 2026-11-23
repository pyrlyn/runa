package runa.engine

import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.json.JsonObject
import runa.core.ChatMessage
import runa.core.ToolCall
import runa.core.UserInputError
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

    /** Rendered with the model's own chat template, which also says how the model calls a tool. */
    public data class Chat(
        val messages: List<ChatMessage>,
        val tools: List<Tool> = emptyList(),
        val toolChoice: ToolChoice = ToolChoice.Auto,
        /** Whether one answer may call several tools; null leaves it to the model's template. */
        val parallelToolCalls: Boolean? = null,
        val format: ResponseFormat = ResponseFormat.Text,
    ) : Prompt {
        /**
         * The tools the model can call: the one [toolChoice] names, all of them, or none. A model that
         * is shown tools it may not call writes the call out as text, so those are not shown at all.
         */
        public fun offeredTools(): List<Tool> =
            when (val choice = toolChoice) {
                is ToolChoice.Named -> {
                    tools.filter { it.name == choice.name }.ifEmpty {
                        throw UserInputError("The tool to call, '${choice.name}', is not among the tools")
                    }
                }

                ToolChoice.Required -> {
                    tools.ifEmpty { throw UserInputError("A tool call is required, but there are no tools") }
                }

                ToolChoice.Auto -> {
                    tools
                }

                ToolChoice.None -> {
                    emptyList()
                }
            }
    }
}

/** A function the model may ask its caller to run. */
public data class Tool(
    val name: String,
    val description: String? = null,
    /** The JSON schema of the arguments; null means the tool takes none. */
    val parameters: JsonObject? = null,
)

public sealed interface ToolChoice {
    /** The model decides between an answer and tool calls. */
    public data object Auto : ToolChoice

    /** The model answers; the tools are only shown to it. */
    public data object None : ToolChoice

    /** The model calls at least one tool. */
    public data object Required : ToolChoice

    /** The model calls this tool. */
    public data class Named(
        val name: String,
    ) : ToolChoice
}

/** What the text of an answer must be. A model cannot leave a required format: it is sampled inside it. */
public sealed interface ResponseFormat {
    public data object Text : ResponseFormat

    /** JSON: any object, or a value of [schema], a JSON schema. */
    public data class Json(
        val schema: JsonObject? = null,
    ) : ResponseFormat
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

    /** The model calls a tool. [index] counts the calls of this answer from 0; the arguments follow. */
    public data class ToolCallStarted(
        val index: Int,
        val id: String,
        val name: String,
    ) : GenerationEvent

    /** The next piece of the arguments of call [index], see [ToolCall.arguments]. */
    public data class ToolCallArguments(
        val index: Int,
        val text: String,
    ) : GenerationEvent

    public data class Finished(
        val reason: FinishReason,
        val stats: GenerationStats,
    ) : GenerationEvent
}

/** @property id the word for it in the APIs and in `--json` output; the OpenAI API uses the same ones. */
public enum class FinishReason(
    public val id: String,
) {
    /** The model ended its turn or produced a stop string. */
    STOP("stop"),

    /** [GenerationRequest.maxTokens] or the context length was reached. */
    LENGTH("length"),

    /** The model ended its turn having called tools: it waits for their results. */
    TOOL_CALLS("tool_calls"),
}

public data class GenerationStats(
    val promptTokens: Int,
    val completionTokens: Int,
    val timeToFirstToken: Duration,
    /** The time from the first token to the last. */
    val generationTime: Duration,
) {
    /** The speed after the first token, which is what [timeToFirstToken] already accounts for. */
    val tokensPerSecond: Double
        get() {
            val seconds = generationTime.toDouble(DurationUnit.SECONDS)
            return if (seconds > 0 && completionTokens > 1) (completionTokens - 1) / seconds else 0.0
        }
}

/** A whole completion, for callers that do not stream. */
public data class Completion(
    val text: String,
    val reason: FinishReason,
    val stats: GenerationStats,
    val toolCalls: List<ToolCall> = emptyList(),
)

/** Puts the events of one generation together into its [Completion]. */
public class CompletionBuilder {
    private val text = StringBuilder()
    private val calls = mutableListOf<GenerationEvent.ToolCallStarted>()
    private val arguments = mutableMapOf<Int, StringBuilder>()
    private var finished: GenerationEvent.Finished? = null

    public fun add(event: GenerationEvent) {
        when (event) {
            is GenerationEvent.Text -> text.append(event.text)
            is GenerationEvent.ToolCallStarted -> calls += event
            is GenerationEvent.ToolCallArguments -> arguments.getOrPut(event.index, ::StringBuilder).append(event.text)
            is GenerationEvent.Finished -> finished = event
        }
    }

    /** @throws IllegalStateException when the generation has not finished. */
    public fun build(): Completion {
        val end = checkNotNull(finished) { "The engine ended generation without a Finished event" }
        val toolCalls = calls.map { ToolCall(it.id, it.name, arguments[it.index]?.toString().orEmpty()) }
        return Completion(text.toString(), end.reason, end.stats, toolCalls)
    }
}

/** Collects a [LoadedModel.generate] flow into one [Completion]. */
public suspend fun Flow<GenerationEvent>.toCompletion(): Completion {
    val completion = CompletionBuilder()
    collect(completion::add)
    return completion.build()
}
