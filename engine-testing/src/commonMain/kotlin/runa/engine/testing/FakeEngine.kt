package runa.engine.testing

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import okio.Path
import runa.core.LoadConfig
import runa.engine.Engine
import runa.engine.EngineInfo
import runa.engine.FinishReason
import runa.engine.GenerationEvent
import runa.engine.GenerationRequest
import runa.engine.GenerationStats
import runa.engine.LoadedModel
import runa.engine.MemoryEstimate
import runa.engine.Prompt
import kotlin.time.Duration.Companion.milliseconds

/**
 * An [Engine] without a model, for the tests of everything that uses one: a token is a word and
 * the completion echoes the prompt. It also shows that the interface does not need llama.cpp.
 */
public class FakeEngine : Engine {
    /** Every load so far, in order. */
    public val loads: MutableList<FakeLoad> = mutableListOf()

    override val info: EngineInfo = EngineInfo(name = "fake", version = "0", accelerators = listOf("cpu"))

    override fun estimateMemory(
        model: Path,
        config: LoadConfig,
    ): MemoryEstimate = MemoryEstimate(weightsBytes = 1_000, contextBytes = (config.contextLength ?: DEFAULT_CONTEXT) * 2L)

    override suspend fun load(
        model: Path,
        config: LoadConfig,
        onProgress: (Float) -> Unit,
    ): LoadedModel {
        onProgress(0f)
        onProgress(1f)
        return FakeModel(config.contextLength ?: DEFAULT_CONTEXT).also { loads += FakeLoad(model, config, it) }
    }

    private companion object {
        const val DEFAULT_CONTEXT = 8
    }
}

public class FakeLoad(
    public val path: Path,
    public val config: LoadConfig,
    public val model: FakeModel,
)

public class FakeModel(
    override val contextLength: Int,
) : LoadedModel {
    public var closed: Boolean = false
        private set
    public var generatedTokens: Int = 0
        private set

    override val supportsEmbeddings: Boolean = true

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
