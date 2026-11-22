package runa.engine.llama

import cnames.structs.llama_context
import cnames.structs.llama_model
import cnames.structs.runa_chat
import cnames.structs.runa_turn
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.toKString
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import runa.core.BackendError
import runa.core.LogLevel
import runa.core.UserInputError
import runa.engine.FinishReason
import runa.engine.GenerationEvent
import runa.engine.GenerationRequest
import runa.engine.GenerationStats
import runa.engine.LoadedModel
import runa.engine.Prompt
import runa.engine.Sampling
import runa.engine.llama.cinterop.LLAMA_DEFAULT_SEED
import runa.engine.llama.cinterop.LLAMA_POOLING_TYPE_NONE
import runa.engine.llama.cinterop.LLAMA_TOKEN_NULL
import runa.engine.llama.cinterop.llama_batch_get_one
import runa.engine.llama.cinterop.llama_decode
import runa.engine.llama.cinterop.llama_encode
import runa.engine.llama.cinterop.llama_free
import runa.engine.llama.cinterop.llama_get_embeddings_seq
import runa.engine.llama.cinterop.llama_get_memory
import runa.engine.llama.cinterop.llama_memory_clear
import runa.engine.llama.cinterop.llama_model_free
import runa.engine.llama.cinterop.llama_model_get_vocab
import runa.engine.llama.cinterop.llama_model_has_decoder
import runa.engine.llama.cinterop.llama_model_n_embd_out
import runa.engine.llama.cinterop.llama_n_batch
import runa.engine.llama.cinterop.llama_n_ctx
import runa.engine.llama.cinterop.llama_n_ubatch
import runa.engine.llama.cinterop.llama_pooling_type
import runa.engine.llama.cinterop.llama_token_to_piece
import runa.engine.llama.cinterop.llama_tokenize
import runa.engine.llama.cinterop.llama_vocab_is_eog
import runa.engine.llama.cinterop.runa_chat_free
import runa.engine.llama.cinterop.runa_chat_init
import runa.engine.llama.cinterop.runa_last_error
import runa.engine.llama.cinterop.runa_sampling
import runa.engine.llama.cinterop.runa_turn_accept
import runa.engine.llama.cinterop.runa_turn_finish
import runa.engine.llama.cinterop.runa_turn_free
import runa.engine.llama.cinterop.runa_turn_init
import runa.engine.llama.cinterop.runa_turn_sample
import runa.engine.llama.cinterop.runa_turn_start
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.math.sqrt
import kotlin.time.Duration
import kotlin.time.TimeSource

/**
 * A model and its context. It owns both native pointers; none leaves this class.
 *
 * Every llama.cpp call runs on [serial], one at a time, and [close] frees the pointers there too.
 * So a call is safe as long as it checks [closed] and uses the pointers without suspending in
 * between: that is what [native] is for.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class LlamaModel(
    private val model: CPointer<llama_model>,
    private val context: CPointer<llama_context>,
    /** The context was created for embeddings and cannot generate. */
    private val embeddings: Boolean,
    private val serial: CoroutineDispatcher,
    private val onLog: (LogLevel, String) -> Unit,
) : LoadedModel {
    private val vocab = checkNotNull(llama_model_get_vocab(model))
    private val batchSize = llama_n_batch(context).toInt()

    // llama.cpp evaluates a batch in pieces of this size; cutting the prompt the same way costs
    // nothing and lets a cancellation in between take effect.
    private val chunkSize = llama_n_ubatch(context).toInt()
    private val closed = AtomicBoolean(false)

    // The context holds one conversation at a time.
    private val busy = Mutex()
    private var pieceBuffer = ByteArray(PIECE_BYTES)

    // Read from the model for its first chat and kept for the next ones.
    private var templates: CPointer<runa_chat>? = null

    override val contextLength: Int = llama_n_ctx(context).toInt()

    override val supportsEmbeddings: Boolean = embeddings && llama_pooling_type(context) != LLAMA_POOLING_TYPE_NONE

    override suspend fun tokenize(text: String): List<Int> {
        checkOpen()
        return withContext(serial) { native { tokens(text, addSpecial = false).toList() } }
    }

    override fun generate(request: GenerationRequest): Flow<GenerationEvent> =
        flow {
            checkOpen()
            busy.withLock { generate(request) }
        }.flowOn(serial)
            // No tokens are computed ahead of the collector, so cancelling it stops generation at once.
            .buffer(Channel.RENDEZVOUS)

    override suspend fun embed(inputs: List<String>): List<FloatArray> {
        checkOpen()
        return busy.withLock {
            withContext(serial) {
                inputs.map { text ->
                    ensureActive()
                    native { embedding(text) }
                }
            }
        }
    }

    override fun close() {
        if (!closed.compareAndSet(expectedValue = false, newValue = true)) return
        runBlocking(serial) {
            templates?.let { runa_chat_free(it) }
            llama_free(context)
            llama_model_free(model)
        }
    }

    private fun checkOpen() = check(!closed.load()) { "The model is closed" }

    /** Runs [block] on llama.cpp's pointers; it must be called on [serial] and must not suspend. */
    private inline fun <T> native(block: () -> T): T {
        checkOpen()
        return block()
    }

    private suspend fun FlowCollector<GenerationEvent>.generate(request: GenerationRequest) {
        val start = TimeSource.Monotonic.markNow()
        val turn =
            native {
                if (embeddings || !llama_model_has_decoder(model)) throw UserInputError("An embedding model cannot generate text")
                turn(request.sampling, request.prompt as? Prompt.Chat)
            }
        try {
            generate(request, turn, start)
        } finally {
            // A turn keeps nothing of the model that freeing it would touch, so this is safe after close().
            runa_turn_free(turn)
        }
    }

    private suspend fun FlowCollector<GenerationEvent>.generate(
        request: GenerationRequest,
        turn: CPointer<runa_turn>,
        start: TimeSource.Monotonic.ValueTimeMark,
    ) {
        val chat = request.prompt is Prompt.Chat
        val opening =
            when (val prompt = request.prompt) {
                is Prompt.Text -> TurnStart(prompt.text, stops = emptyList())
                is Prompt.Chat -> native { turnStart(checkNotNull(runa_turn_start(turn)).toKString()) }
            }
        val prompt = native { tokens(opening.prompt, addSpecial = true) }
        if (prompt.isEmpty()) throw UserInputError("The prompt is empty")
        if (prompt.size >= contextLength) {
            throw UserInputError(
                "The prompt is ${prompt.size} tokens long, but the model was loaded with a context of $contextLength tokens",
            )
        }

        // Every call starts a new conversation: nothing of the previous one may leak into it.
        native { llama_memory_clear(llama_get_memory(context), true) }
        for (from in prompt.indices step chunkSize) {
            currentCoroutineContext().ensureActive()
            native { decode(prompt, from, minOf(chunkSize, prompt.size - from)) }
        }

        val limit = minOf(request.maxTokens ?: Int.MAX_VALUE, contextLength - prompt.size)
        val text = Utf8Assembler()
        val stop = StopFilter(request.stop + opening.stops)
        val token = IntArray(1)
        var produced = 0
        var firstToken: Duration? = null
        var ended = false
        var called = false

        suspend fun deliver(events: List<GenerationEvent>) {
            for (event in events) {
                if (stop.stopped) return
                if (event is GenerationEvent.Text) {
                    emitText(stop.append(event.text))
                } else {
                    // The model wrote the text held back as a possible stop string before this call.
                    emitText(stop.flush())
                    emit(event)
                    called = true
                }
            }
        }

        while (produced < limit) {
            currentCoroutineContext().ensureActive()
            val events =
                native {
                    token[0] = runa_turn_sample(turn, context)
                    if (token[0] == LLAMA_TOKEN_NULL) throw BackendError("llama.cpp cannot choose a token: ${lastError()}")
                    when {
                        llama_vocab_is_eog(vocab, token[0]) -> null

                        // A chat's answer is read by the parser of its template: it tells text from tool calls.
                        chat -> turnEvents(reported(runa_turn_accept(turn, token[0])))

                        else -> listOf(GenerationEvent.Text(text.append(piece(token[0]))))
                    }
                }
            if (firstToken == null) firstToken = start.elapsedNow()
            if (events == null) {
                ended = true
                break
            }
            produced++
            deliver(events)
            if (stop.stopped) break
            // The last token is never evaluated: nothing would read its result.
            if (produced < limit) native { decode(token, 0, 1) }
        }
        if (chat) {
            if (!stop.stopped) deliver(native { turnEvents(reported(runa_turn_finish(turn, ended))) })
        } else {
            emitText(stop.append(text.flush()))
        }
        emitText(stop.flush())

        val reason =
            when {
                !ended && !stop.stopped -> FinishReason.LENGTH
                called -> FinishReason.TOOL_CALLS
                else -> FinishReason.STOP
            }
        val total = start.elapsedNow()
        val toFirst = firstToken ?: total
        emit(GenerationEvent.Finished(reason, GenerationStats(prompt.size, produced, toFirst, total - toFirst)))
    }

    private suspend fun FlowCollector<GenerationEvent>.emitText(text: String) {
        if (text.isNotEmpty()) emit(GenerationEvent.Text(text))
    }

    private fun tokens(
        text: String,
        addSpecial: Boolean,
    ): IntArray {
        val bytes = text.encodeToByteArray().size
        // A token is at least one byte long; the rest is room for BOS and EOS.
        var tokens = IntArray(bytes + SPECIAL_TOKENS)

        // Special tokens written out in the text (`<|im_start|>`) are read as tokens: the chat template relies on it.
        fun tokenize() = tokens.usePinned { llama_tokenize(vocab, text, bytes, it.addressOf(0), tokens.size, addSpecial, true) }
        var count = tokenize()
        if (count < 0) {
            tokens = IntArray(-count)
            count = tokenize()
        }
        if (count < 0) throw BackendError("llama.cpp cannot tokenize the text")
        return tokens.copyOf(count)
    }

    /** The bytes of [token]; control tokens have none. */
    private fun piece(token: Int): ByteArray {
        fun write() = pieceBuffer.usePinned { llama_token_to_piece(vocab, token, it.addressOf(0), pieceBuffer.size, 0, false) }
        var length = write()
        if (length < 0) {
            pieceBuffer = ByteArray(-length)
            length = write()
        }
        return pieceBuffer.copyOf(maxOf(length, 0))
    }

    private fun decode(
        tokens: IntArray,
        from: Int,
        count: Int,
    ) {
        val status = tokens.usePinned { llama_decode(context, llama_batch_get_one(it.addressOf(from), count)) }
        if (status != 0) throw BackendError("llama.cpp failed to evaluate $count tokens (llama_decode returned $status)")
    }

    /**
     * The sampler of one answer and, for a chat, its prompt and the parser of what the model writes.
     * The caller frees it.
     */
    private fun turn(
        sampling: Sampling,
        chat: Prompt.Chat?,
    ): CPointer<runa_turn> =
        memScoped {
            val settings =
                alloc<runa_sampling>().apply {
                    temperature = sampling.temperature.toFloat()
                    top_p = sampling.topP.toFloat()
                    min_p = sampling.minP.toFloat()
                    repeat_penalty = sampling.repeatPenalty.toFloat()
                    top_k = sampling.topK
                    seed = sampling.seed?.toUInt() ?: LLAMA_DEFAULT_SEED
                }
            if (chat == null) {
                runa_turn_init(model, null, null, settings.ptr) ?: throw BackendError("llama.cpp cannot create a sampler: ${lastError()}")
            } else {
                // What llama.cpp refuses here is the request: messages the template does not take,
                // a tool or a schema it cannot turn into a grammar.
                runa_turn_init(model, templates(), chatRequest(chat).toString(), settings.ptr) ?: throw UserInputError(lastError())
            }
        }

    /** The model's chat template. One that llama.cpp cannot read is replaced by ChatML, the most common format. */
    private fun templates(): CPointer<runa_chat> =
        templates ?: run {
            runa_chat_init(model, null)
                ?: run {
                    onLog(LogLevel.WARN, "llama.cpp cannot read this model's chat template (${lastError()}); using ChatML")
                    runa_chat_init(model, FALLBACK_TEMPLATE)
                }
                ?: throw BackendError("llama.cpp cannot render a chat: ${lastError()}")
        }.also { templates = it }

    private fun lastError(): String = runa_last_error()?.toKString().orEmpty()

    /** The JSON a call of the shim returned, or the error it failed with. */
    private fun reported(json: CPointer<ByteVar>?): String =
        json?.toKString() ?: throw BackendError("llama.cpp cannot read the model's answer: ${lastError()}")

    /** The pooled embedding of [text], scaled to length 1. */
    private fun embedding(text: String): FloatArray {
        if (!supportsEmbeddings) throw UserInputError("This model does not produce embeddings")
        val tokens = tokens(text, addSpecial = true)
        if (tokens.isEmpty()) throw UserInputError("Cannot embed an empty text")
        if (tokens.size > batchSize) throw UserInputError("The text is ${tokens.size} tokens long, but the model embeds at most $batchSize")

        llama_memory_clear(llama_get_memory(context), true)
        val status =
            tokens.usePinned {
                val batch = llama_batch_get_one(it.addressOf(0), tokens.size)
                if (llama_model_has_decoder(model)) llama_decode(context, batch) else llama_encode(context, batch)
            }
        if (status != 0) throw BackendError("llama.cpp failed to embed the text (status $status)")
        val values = llama_get_embeddings_seq(context, 0) ?: throw BackendError("llama.cpp returned no embedding")
        val vector = FloatArray(llama_model_n_embd_out(model)) { values[it] }
        val norm = sqrt(vector.sumOf { it.toDouble() * it }).toFloat()
        return if (norm > 0f) FloatArray(vector.size) { vector[it] / norm } else vector
    }

    private companion object {
        const val PIECE_BYTES = 64
        const val SPECIAL_TOKENS = 8
        const val FALLBACK_TEMPLATE = "chatml"
    }
}
