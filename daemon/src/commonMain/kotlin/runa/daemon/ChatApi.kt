package runa.daemon

import io.ktor.server.application.ApplicationCall
import kotlinx.serialization.SerializationStrategy
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import runa.core.ChatMessage
import runa.core.ChatRole
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.ChatDelta
import runa.core.api.ChatEnded
import runa.core.api.ChatFailed
import runa.core.api.ChatMark
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.ChatStarted
import runa.core.api.ChatStats
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.engine.Completion
import runa.engine.GenerationEvent
import runa.engine.GenerationRequest
import runa.engine.Prompt
import runa.engine.toCompletion
import kotlin.math.roundToLong
import kotlin.time.DurationUnit

/**
 * `POST /api/v1/chat`, which `runa chat` talks to: a conversation that the daemon remembers, so
 * each request carries only what the user said next.
 */
internal class ChatApi(
    private val registry: ModelRegistry,
    private val failures: Failures,
    private val log: DaemonLog,
    private val conversations: Conversations = Conversations(),
    /** The random part of a response id. */
    private val newId: () -> String = ::randomHex,
) {
    suspend fun answer(
        call: ApplicationCall,
        request: ChatRequest,
    ) {
        if (request.integrations.hasItems()) throw UserInputError("integrations are not supported yet")
        val input = text(request.input)
        if (input.isBlank()) throw UserInputError("input must not be empty")
        val previous =
            request.previousResponseId?.let {
                conversations.find(it)
                    ?: throw UserInputError("No response is called '$it'; the daemon keeps the last ones in memory only")
            }
        val systemPrompt = (request.systemPrompt ?: previous?.systemPrompt)?.takeIf { it.isNotBlank() }
        val messages = previous?.messages.orEmpty() + ChatMessage(ChatRole.USER, input)
        val generation =
            GenerationRequest(
                prompt = Prompt.Chat(listOfNotNull(systemPrompt?.let { ChatMessage(ChatRole.SYSTEM, it) }) + messages),
                sampling = samplingOf(request.temperature, request.topP, request.topK, request.minP, request.repeatPenalty),
                maxTokens = tokenLimit(request.maxOutputTokens, "max_output_tokens"),
            )
        registry.useNamed(request.model) { id, model ->
            val turn = Turn(id, Conversation(systemPrompt, messages), request.store)
            if (request.stream) {
                call.respondEvents(model.generate(generation), turn, failures)
            } else {
                val result = turn.result(model.generate(generation).toCompletion())
                call.respondJson(DaemonJson, ChatResponse.serializer(), result)
            }
        }
    }

    private fun text(input: JsonElement): String =
        when (input) {
            is JsonArray -> {
                input.joinToString("\n") { part ->
                    val type = ((part as? JsonObject)?.get("type") as? JsonPrimitive)?.content
                    if (type != TEXT) throw UserInputError("Only text input is supported, not '$type'")
                    string(part["content"]) ?: throw UserInputError("A text input needs its content")
                }
            }

            else -> {
                string(input) ?: throw UserInputError("input must be a string or a list of text parts")
            }
        }

    private fun string(value: JsonElement?): String? = (value as? JsonPrimitive)?.takeIf { it.isString }?.content

    /** One request on the instance [model]: [asked] is the conversation with the user's new message. */
    private inner class Turn(
        private val model: String,
        private val asked: Conversation,
        private val store: Boolean,
    ) : EventFormat {
        private val answer = StringBuilder()

        override fun opening() =
            listOf(
                frame(DaemonRoutes.EVENT_CHAT_START, ChatStarted.serializer(), ChatStarted(modelInstanceId = model)),
                frame(DaemonRoutes.EVENT_MESSAGE_START, ChatMark.serializer(), ChatMark(DaemonRoutes.EVENT_MESSAGE_START)),
            )

        override fun text(text: String): List<String> {
            answer.append(text)
            return listOf(frame(DaemonRoutes.EVENT_MESSAGE_DELTA, ChatDelta.serializer(), ChatDelta(content = text)))
        }

        override suspend fun finished(event: GenerationEvent.Finished) =
            listOf(
                frame(DaemonRoutes.EVENT_MESSAGE_END, ChatMark.serializer(), ChatMark(DaemonRoutes.EVENT_MESSAGE_END)),
                frame(
                    DaemonRoutes.EVENT_CHAT_END,
                    ChatEnded.serializer(),
                    ChatEnded(result = result(Completion(answer.toString(), event.reason, event.stats))),
                ),
            )

        override fun failed(error: ApiError) =
            listOf(frame(DaemonRoutes.EVENT_ERROR, ChatFailed.serializer(), ChatFailed(error = error.error)))

        /** Remembers the answer, unless the request said not to, and describes it. */
        suspend fun result(completion: Completion): ChatResponse {
            val responseId =
                if (store) {
                    val answered = asked.messages + ChatMessage(ChatRole.ASSISTANT, completion.text)
                    "$RESPONSE_PREFIX${newId()}".also { conversations.keep(it, Conversation(asked.systemPrompt, answered)) }
                } else {
                    null
                }
            val stats = completion.stats
            log.report("chat", model, stats)
            return ChatResponse(
                modelInstanceId = model,
                output = listOf(ChatResponse.Output(ChatResponse.MESSAGE, completion.text)),
                stats =
                    ChatStats(
                        inputTokens = stats.promptTokens,
                        totalOutputTokens = stats.completionTokens,
                        tokensPerSecond = rounded(stats.tokensPerSecond, 100),
                        timeToFirstTokenSeconds = rounded(stats.timeToFirstToken.toDouble(DurationUnit.SECONDS), 1_000),
                        stopReason = completion.reason.id,
                    ),
                responseId = responseId,
            )
        }

        private fun <T> frame(
            event: String,
            serializer: SerializationStrategy<T>,
            value: T,
        ) = sseFrame(DaemonJson.encodeToString(serializer, value), event)
    }

    private companion object {
        const val TEXT = "text"
        const val RESPONSE_PREFIX = "resp_"

        /** More digits than this are noise, and fewer keep the JSON readable. */
        fun rounded(
            value: Double,
            steps: Int,
        ) = (value * steps).roundToLong() / steps.toDouble()
    }
}
