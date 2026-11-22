package runa.engine

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import okio.Path
import runa.core.LoadConfig
import kotlin.time.Duration.Companion.milliseconds

/**
 * An [Engine] without a model: a token is a word and the completion echoes the prompt. It shows
 * that the interface can be implemented and used without llama.cpp.
 */
internal class FakeEngine : Engine {
    val loaded = mutableListOf<FakeModel>()

    override val info = EngineInfo(name = "fake", version = "0", accelerators = listOf("cpu"))

    override fun estimateMemory(
        model: Path,
        config: LoadConfig,
    ) = MemoryEstimate(weightsBytes = 1_000, contextBytes = (config.contextLength ?: DEFAULT_CONTEXT) * 2L)

    override suspend fun load(
        model: Path,
        config: LoadConfig,
        onProgress: (Float) -> Unit,
    ): LoadedModel {
        onProgress(0f)
        onProgress(1f)
        return FakeModel(config.contextLength ?: DEFAULT_CONTEXT).also(loaded::add)
    }

    private companion object {
        const val DEFAULT_CONTEXT = 8
    }
}

internal class FakeModel(
    override val contextLength: Int,
) : LoadedModel {
    var closed = false
        private set
    var generatedTokens = 0
        private set

    override val supportsEmbeddings = true

    override suspend fun tokenize(text: String): List<Int> = words(text).map { it.hashCode() }

    override fun generate(request: GenerationRequest): Flow<GenerationEvent> =
        flow {
            checkOpen()
            val prompt =
                when (val prompt = request.prompt) {
                    is Prompt.Text -> prompt.text
                    is Prompt.Chat -> prompt.messages.joinToString(" ") { "${it.role.id}: ${it.content}" }
                }
            val words = words(prompt)
            val limit = minOf(request.maxTokens ?: Int.MAX_VALUE, contextLength)
            var emitted = 0
            var reason = FinishReason.STOP
            for (word in words) {
                if (word in request.stop) break
                if (emitted == limit) {
                    reason = FinishReason.LENGTH
                    break
                }
                emit(GenerationEvent.Text(if (emitted == 0) word else " $word"))
                emitted++
                generatedTokens++
            }
            emit(GenerationEvent.Finished(reason, GenerationStats(words.size, emitted, 10.milliseconds, (emitted * 100).milliseconds)))
        }

    override suspend fun embed(inputs: List<String>): List<FloatArray> {
        checkOpen()
        return inputs.map { floatArrayOf(it.length.toFloat()) }
    }

    override fun close() {
        closed = true
    }

    private fun checkOpen() = check(!closed) { "The model is closed" }

    private fun words(text: String) = text.split(' ').filter { it.isNotEmpty() }
}
