package runa.engine.llama

import cnames.structs.llama_context
import cnames.structs.llama_model
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.cstr
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.readBytes
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
import runa.core.ChatMessage
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
import runa.engine.llama.cinterop.llama_batch_get_one
import runa.engine.llama.cinterop.llama_chat_apply_template
import runa.engine.llama.cinterop.llama_chat_message
import runa.engine.llama.cinterop.llama_decode
import runa.engine.llama.cinterop.llama_encode
import runa.engine.llama.cinterop.llama_free
import runa.engine.llama.cinterop.llama_get_embeddings_seq
import runa.engine.llama.cinterop.llama_get_memory
import runa.engine.llama.cinterop.llama_memory_clear
import runa.engine.llama.cinterop.llama_model_chat_template
import runa.engine.llama.cinterop.llama_model_free
import runa.engine.llama.cinterop.llama_model_get_vocab
import runa.engine.llama.cinterop.llama_model_has_decoder
import runa.engine.llama.cinterop.llama_model_n_embd_out
import runa.engine.llama.cinterop.llama_n_batch
import runa.engine.llama.cinterop.llama_n_ctx
import runa.engine.llama.cinterop.llama_n_ubatch
import runa.engine.llama.cinterop.llama_pooling_type
import runa.engine.llama.cinterop.llama_sampler
import runa.engine.llama.cinterop.llama_sampler_chain_add
import runa.engine.llama.cinterop.llama_sampler_chain_default_params
import runa.engine.llama.cinterop.llama_sampler_chain_init
import runa.engine.llama.cinterop.llama_sampler_free
import runa.engine.llama.cinterop.llama_sampler_init_dist
import runa.engine.llama.cinterop.llama_sampler_init_greedy
import runa.engine.llama.cinterop.llama_sampler_init_min_p
import runa.engine.llama.cinterop.llama_sampler_init_penalties
import runa.engine.llama.cinterop.llama_sampler_init_temp
import runa.engine.llama.cinterop.llama_sampler_init_top_k
import runa.engine.llama.cinterop.llama_sampler_init_top_p
import runa.engine.llama.cinterop.llama_sampler_sample
import runa.engine.llama.cinterop.llama_token_to_piece
import runa.engine.llama.cinterop.llama_tokenize
import runa.engine.llama.cinterop.llama_vocab_is_eog
import runa.engine.llama.cinterop.llama_vocab_n_tokens
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
        val prompt =
            native {
                if (embeddings || !llama_model_has_decoder(model)) throw UserInputError("An embedding model cannot generate text")
                when (val prompt = request.prompt) {
                    is Prompt.Text -> tokens(prompt.text, addSpecial = true)
                    is Prompt.Chat -> tokens(render(prompt.messages), addSpecial = true)
                }
            }
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
        val stop = StopFilter(request.stop)
        val token = IntArray(1)
        var produced = 0
        var firstToken: Duration? = null
        var reason = FinishReason.LENGTH
        val sampler = native { sampler(request.sampling) }
        try {
            while (produced < limit) {
                currentCoroutineContext().ensureActive()
                val piece =
                    native {
                        token[0] = llama_sampler_sample(sampler, context, LAST_OUTPUT)
                        if (llama_vocab_is_eog(vocab, token[0])) null else piece(token[0])
                    }
                if (firstToken == null) firstToken = start.elapsedNow()
                if (piece == null) {
                    reason = FinishReason.STOP
                    break
                }
                produced++
                emitText(stop.append(text.append(piece)))
                if (stop.stopped) {
                    reason = FinishReason.STOP
                    break
                }
                // The last token is never evaluated: nothing would read its result.
                if (produced < limit) native { decode(token, 0, 1) }
            }
        } finally {
            // A sampler does not refer to the model, so this is safe after close().
            llama_sampler_free(sampler)
        }
        emitText(stop.append(text.flush()) + stop.flush())

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
     * The chat as one prompt, ending where the assistant's answer starts. llama.cpp recognises the
     * common templates by their text and has no Jinja interpreter in its C API, so a model with a
     * template it does not know is prompted with ChatML.
     */
    private fun render(messages: List<ChatMessage>): String {
        if (messages.isEmpty()) throw UserInputError("A chat needs at least one message")
        val template = llama_model_chat_template(model, null)?.toKString()
        return memScoped {
            val chat = allocArray<llama_chat_message>(messages.size)
            messages.forEachIndexed { index, message ->
                chat[index].role = message.role.id.cstr.ptr
                chat[index].content = message.content.cstr.ptr
            }

            fun apply(template: String): String? {
                var size = 2 * messages.sumOf { it.content.encodeToByteArray().size } + TEMPLATE_BYTES
                while (true) {
                    val buffer = allocArray<ByteVar>(size)
                    val written = llama_chat_apply_template(template, chat, messages.size.toULong(), true, buffer, size)
                    if (written < 0) return null
                    if (written <= size) return buffer.readBytes(written).decodeToString()
                    size = written
                }
            }
            template?.let(::apply) ?: run {
                if (template != null) onLog(LogLevel.WARN, "llama.cpp does not know this model's chat template; using ChatML")
                apply(FALLBACK_TEMPLATE) ?: throw BackendError("llama.cpp cannot render the chat")
            }
        }
    }

    /** llama.cpp's default sampler order; temperature 0 takes the most likely token and needs none of it. */
    private fun sampler(sampling: Sampling): CPointer<llama_sampler> {
        val chain = checkNotNull(llama_sampler_chain_init(llama_sampler_chain_default_params()))

        fun add(sampler: CPointer<llama_sampler>?) = llama_sampler_chain_add(chain, sampler)
        if (sampling.repeatPenalty != 1.0) {
            add(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), PENALTY_WINDOW, sampling.repeatPenalty.toFloat(), 0f, 0f))
        }
        if (sampling.temperature <= 0.0) {
            add(llama_sampler_init_greedy())
        } else {
            add(llama_sampler_init_top_k(sampling.topK))
            add(llama_sampler_init_top_p(sampling.topP.toFloat(), 1u))
            add(llama_sampler_init_min_p(sampling.minP.toFloat(), 1u))
            add(llama_sampler_init_temp(sampling.temperature.toFloat()))
            add(llama_sampler_init_dist(sampling.seed?.toUInt() ?: LLAMA_DEFAULT_SEED))
        }
        return chain
    }

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
        const val TEMPLATE_BYTES = 1024

        /** `llama_sampler_sample`'s index of the last evaluated token. */
        const val LAST_OUTPUT = -1

        /** llama.cpp's own default for how many recent tokens the repeat penalty looks at. */
        const val PENALTY_WINDOW = 64
        const val FALLBACK_TEMPLATE = "chatml"
    }
}
