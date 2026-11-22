package runa.daemon

import io.ktor.http.ContentType
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respondText
import io.ktor.server.routing.Routing
import io.ktor.server.routing.RoutingContext
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import runa.core.ChatMessage
import runa.core.ChatRole
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.DaemonJson
import runa.engine.Completion
import runa.engine.FinishReason
import runa.engine.GenerationEvent
import runa.engine.GenerationRequest
import runa.engine.GenerationStats
import runa.engine.LoadedModel
import runa.engine.Prompt
import runa.engine.toCompletion
import runa.store.StoreListing
import kotlin.io.encoding.Base64
import kotlin.time.Clock

/**
 * The OpenAI-compatible API: what programs written for OpenAI's API need to talk to a local model.
 * A request names its model as `runa ls` or `runa ps` shows it; a model that is not loaded yet is
 * loaded by the first request for it.
 */
internal class OpenAiApi(
    private val registry: ModelRegistry,
    private val store: () -> StoreListing,
    private val access: Access,
    private val failures: Failures,
    private val log: DaemonLog,
    /** Seconds since 1970, for the `created` fields. */
    private val now: () -> Long = { Clock.System.now().epochSeconds },
    /** The random part of a completion's id. */
    private val newId: () -> String = ::randomHex,
) {
    // A stored model has no creation time that would mean anything to a client.
    private val started = now()

    fun install(
        routing: Routing,
        listener: Listener,
    ) = with(routing) {
        get(OpenAiRoutes.MODELS) {
            serve(listener) { call.respondJson(OpenAiJson, ModelCardList.serializer(), models()) }
        }
        post(OpenAiRoutes.CHAT_COMPLETIONS) {
            serve(listener) { chat(call, call.receiveJson(OpenAiJson, ChatCompletionRequest.serializer())) }
        }
        post(OpenAiRoutes.COMPLETIONS) {
            serve(listener) { complete(call, call.receiveJson(OpenAiJson, CompletionRequest.serializer())) }
        }
        post(OpenAiRoutes.EMBEDDINGS) {
            serve(listener) { embed(call, call.receiveJson(OpenAiJson, EmbeddingRequest.serializer())) }
        }
    }

    private suspend fun RoutingContext.serve(
        listener: Listener,
        block: suspend () -> Unit,
    ) = failures.answer(call) {
        access.requireInference(call, listener)
        block()
    }

    /** The loaded instances with a name of their own, then every stored model. */
    private suspend fun models(): ModelCardList {
        fun card(
            id: String,
            owner: String,
        ) = ModelCard(id, "model", started, owner)
        val stored = store().models
        val keys = stored.map { it.key.toString() }.toSet()
        val named = registry.instances().filter { it.id !in keys }
        return ModelCardList(
            "list",
            named.map { card(it.id, it.model.substringBefore('/')) } + stored.map { card(it.key.toString(), it.key.publisher) },
        )
    }

    private suspend fun chat(
        call: ApplicationCall,
        request: ChatCompletionRequest,
    ) {
        if (request.tools.hasItems() || request.functions.hasItems()) throw UserInputError("tools are not supported yet")
        val format = (request.responseFormat as? JsonObject)?.get("type")?.let { string(it, "response_format.type") }
        if (format != null && format != "text") throw UserInputError("response_format '$format' is not supported yet")
        requireOneChoice(request.n)
        if (request.messages.isEmpty()) throw UserInputError("messages must not be empty")
        val generation =
            GenerationRequest(
                prompt = Prompt.Chat(request.messages.map(::chatMessage)),
                sampling =
                    samplingOf(request.temperature, request.topP, request.topK, request.minP, request.repeatPenalty, request.seed),
                maxTokens = tokenLimit(request.maxCompletionTokens ?: request.maxTokens, "max_tokens"),
                stop = strings(request.stop, "stop"),
            )
        registry.useNamed(request.model) { id, model ->
            generate(call, model, generation, request.stream, request.streamOptions, ChatShape("chatcmpl-${newId()}", now(), id))
        }
    }

    private suspend fun complete(
        call: ApplicationCall,
        request: CompletionRequest,
    ) {
        requireOneChoice(request.n)
        if (request.echo) throw UserInputError("echo is not supported")
        if (request.suffix != null) throw UserInputError("suffix is not supported")
        val prompt = strings(request.prompt, "prompt").singleOrNull() ?: throw UserInputError("prompt must be one string")
        val generation =
            GenerationRequest(
                prompt = Prompt.Text(prompt),
                sampling =
                    samplingOf(request.temperature, request.topP, request.topK, request.minP, request.repeatPenalty, request.seed),
                maxTokens = tokenLimit(request.maxTokens, "max_tokens"),
                stop = strings(request.stop, "stop"),
            )
        registry.useNamed(request.model) { id, model ->
            generate(call, model, generation, request.stream, request.streamOptions, TextShape("cmpl-${newId()}", now(), id))
        }
    }

    private suspend fun generate(
        call: ApplicationCall,
        model: LoadedModel,
        generation: GenerationRequest,
        stream: Boolean,
        options: StreamOptions?,
        shape: Shape,
    ) {
        if (!stream) {
            val completion = model.generate(generation).toCompletion()
            call.respondText(shape.whole(completion), ContentType.Application.Json)
            log.report(shape.name, shape.model, completion.stats)
            return
        }
        call.respondEvents(model.generate(generation), Events(shape, options), failures)
    }

    private suspend fun embed(
        call: ApplicationCall,
        request: EmbeddingRequest,
    ) {
        val inputs = strings(request.input, "input")
        if (inputs.isEmpty()) throw UserInputError("input must not be empty")
        if (request.dimensions != null) throw UserInputError("dimensions is not supported: the model decides the size of its embeddings")
        val encode: (FloatArray) -> JsonElement =
            when (request.encodingFormat) {
                EmbeddingRequest.FLOAT -> { vector -> JsonArray(vector.map(::JsonPrimitive)) }
                EmbeddingRequest.BASE64 -> ::base64
                else -> throw UserInputError("encoding_format must be float or base64")
            }
        registry.useNamed(request.model) { id, model ->
            if (!model.supportsEmbeddings) throw UserInputError("'$id' is not an embedding model")
            val vectors = model.embed(inputs)
            val tokens = inputs.sumOf { model.tokenize(it).size }
            val data = vectors.mapIndexed { index, vector -> Embedding("embedding", index, encode(vector)) }
            call.respondJson(OpenAiJson, EmbeddingList.serializer(), EmbeddingList("list", data, id, EmbeddingUsage(tokens, tokens)))
            log.info("embeddings '$id': ${inputs.size} inputs, $tokens tokens")
        }
    }

    private fun chatMessage(message: ChatCompletionRequest.Message): ChatMessage {
        val role =
            when (message.role) {
                // OpenAI's newer name for the system prompt.
                "developer" -> ChatRole.SYSTEM

                else -> ChatRole.entries.firstOrNull { it.id == message.role }
            } ?: throw UserInputError("'${message.role}' is not a message role")
        return ChatMessage(role, text(message.content))
    }

    private fun text(content: JsonElement?): String =
        when (content) {
            // An assistant message that only called tools has no content.
            null, JsonNull -> {
                ""
            }

            is JsonArray -> {
                content.joinToString("") { part ->
                    val fields = part as? JsonObject ?: throw UserInputError("content must be a string or a list of parts")
                    val type = fields["type"]?.let { string(it, "content type") }
                    if (type != "text") throw UserInputError("Only text content is supported, not '$type'")
                    string(fields["text"] ?: JsonNull, "content text")
                }
            }

            else -> {
                string(content, "content")
            }
        }

    private fun requireOneChoice(n: Int?) {
        if (n != null && n != 1) throw UserInputError("n must be 1")
    }

    private fun strings(
        value: JsonElement?,
        field: String,
    ): List<String> =
        when (value) {
            null, JsonNull -> emptyList()
            is JsonArray -> value.map { string(it, field) }
            else -> listOf(string(value, field))
        }

    private fun string(
        value: JsonElement,
        field: String,
    ): String =
        (value as? JsonPrimitive)?.takeIf { it.isString }?.content ?: throw UserInputError("$field must be a string or a list of strings")

    private fun base64(vector: FloatArray): JsonElement {
        val bytes = ByteArray(vector.size * Float.SIZE_BYTES)
        vector.forEachIndexed { index, value ->
            val bits = value.toRawBits()
            // Little-endian, which is what the OpenAI clients decode.
            repeat(Float.SIZE_BYTES) { bytes[index * Float.SIZE_BYTES + it] = (bits ushr (it * Byte.SIZE_BITS)).toByte() }
        }
        return JsonPrimitive(Base64.encode(bytes))
    }

    /** A stream of `data:` events, ended by [DONE] when the generation was. */
    private inner class Events(
        private val shape: Shape,
        private val options: StreamOptions?,
    ) : EventFormat {
        override fun opening() = frames(shape.opening())

        override fun text(text: String) = frames(shape.text(text))

        override suspend fun finished(event: GenerationEvent.Finished): List<String> {
            log.report(shape.name, shape.model, event.stats)
            return frames(shape.closing(event.reason), shape.usage(event.stats).takeIf { options?.includeUsage == true }, DONE)
        }

        // As OpenAI sends a failure: the last event, and no [DONE] after it.
        override fun failed(error: ApiError) = frames(DaemonJson.encodeToString(ApiError.serializer(), error))

        private fun frames(vararg data: String?) = data.filterNotNull().map { sseFrame(it) }
    }

    /** How one endpoint writes a generation: as a whole, or as the events of a stream. */
    private interface Shape {
        /** The endpoint, for the log. */
        val name: String
        val model: String

        fun whole(completion: Completion): String

        /** The event before any text, if the format has one. */
        fun opening(): String?

        fun text(text: String): String

        fun closing(reason: FinishReason): String

        /** The extra last event of `stream_options.include_usage`. */
        fun usage(stats: GenerationStats): String
    }

    private class ChatShape(
        private val id: String,
        private val created: Long,
        override val model: String,
    ) : Shape {
        override val name = "chat.completions"

        override fun whole(completion: Completion): String {
            val choice = ChatCompletion.Choice(0, ChatCompletion.Message(ChatRole.ASSISTANT.id, completion.text), completion.reason.id)
            val answer = ChatCompletion(id, "chat.completion", created, model, listOf(choice), usageOf(completion.stats))
            return OpenAiJson.encodeToString(ChatCompletion.serializer(), answer)
        }

        override fun opening() = chunk(ChatCompletionChunk.Delta(role = ChatRole.ASSISTANT.id, content = ""))

        override fun text(text: String) = chunk(ChatCompletionChunk.Delta(content = text))

        override fun closing(reason: FinishReason) = chunk(ChatCompletionChunk.Delta(), reason.id)

        override fun usage(stats: GenerationStats) = encode(emptyList(), usageOf(stats))

        private fun chunk(
            delta: ChatCompletionChunk.Delta,
            finishReason: String? = null,
        ) = encode(listOf(ChatCompletionChunk.Choice(0, delta, finishReason)), usage = null)

        private fun encode(
            choices: List<ChatCompletionChunk.Choice>,
            usage: Usage?,
        ) = OpenAiJson.encodeToString(
            ChatCompletionChunk.serializer(),
            ChatCompletionChunk(id, "chat.completion.chunk", created, model, choices, usage),
        )
    }

    private class TextShape(
        private val id: String,
        private val created: Long,
        override val model: String,
    ) : Shape {
        override val name = "completions"

        override fun whole(completion: Completion) =
            encode(listOf(TextCompletion.Choice(0, completion.text, completion.reason.id)), usageOf(completion.stats))

        override fun opening(): String? = null

        override fun text(text: String) = encode(listOf(TextCompletion.Choice(0, text, null)))

        override fun closing(reason: FinishReason) = encode(listOf(TextCompletion.Choice(0, "", reason.id)))

        override fun usage(stats: GenerationStats) = encode(emptyList(), usageOf(stats))

        private fun encode(
            choices: List<TextCompletion.Choice>,
            usage: Usage? = null,
        ) = OpenAiJson.encodeToString(TextCompletion.serializer(), TextCompletion(id, "text_completion", created, model, choices, usage))
    }

    private companion object {
        const val DONE = "[DONE]"

        fun usageOf(stats: GenerationStats) = Usage(stats.promptTokens, stats.completionTokens, stats.promptTokens + stats.completionTokens)
    }
}
