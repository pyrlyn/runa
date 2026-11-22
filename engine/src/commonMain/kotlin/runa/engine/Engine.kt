package runa.engine

import kotlinx.coroutines.flow.Flow
import okio.Path
import runa.core.LoadConfig

/**
 * An inference backend: it turns a model file into a [LoadedModel]. llama.cpp is the first
 * implementation; the daemon, `runa run` and the tests only ever see this interface, so no other
 * module depends on a backend's C API.
 *
 * Implementations are safe to call from any coroutine. A backend whose native state is bound to
 * one thread (llama.cpp) confines its calls to its own dispatcher.
 */
public interface Engine {
    public val info: EngineInfo

    /** What loading [model] with [config] would take, from the file's header alone. Loads nothing. */
    public fun estimateMemory(
        model: Path,
        config: LoadConfig,
    ): MemoryEstimate

    /**
     * Loads [model]. [onProgress] gets values from 0 to 1 and may be called from another thread.
     * Cancelling the coroutine aborts the load and frees what was allocated. An engine may load
     * with another context length than [config] asks for (no more than the model supports,
     * rounded to what the backend allocates); [LoadedModel.contextLength] has the real one.
     *
     * @throws runa.core.UserInputError when the file is missing or is not a model this engine can run.
     */
    public suspend fun load(
        model: Path,
        config: LoadConfig,
        onProgress: (Float) -> Unit = {},
    ): LoadedModel
}

public data class EngineInfo(
    /** A stable identifier, for example `llama.cpp`. */
    val name: String,
    /** The backend's own version, for example a llama.cpp tag. */
    val version: String,
    /** The compute backends compiled in, for example `cpu` and `metal`. */
    val accelerators: List<String>,
)

public data class MemoryEstimate(
    val weightsBytes: Long,
    /** The KV cache and compute buffers for the requested context length. */
    val contextBytes: Long,
) {
    val totalBytes: Long get() = weightsBytes + contextBytes
}

/**
 * A model in memory. It owns native resources: [close] frees them, and every other call after
 * [close] throws [IllegalStateException]. Closing twice is allowed.
 */
public interface LoadedModel : AutoCloseable {
    /** The context length the model was actually loaded with. */
    public val contextLength: Int

    public val supportsEmbeddings: Boolean

    public suspend fun tokenize(text: String): List<Int>

    /**
     * Generates a completion as a cold flow: nothing runs until it is collected, and cancelling
     * the collector stops generation between two tokens. Unless cancelled, the flow ends with
     * exactly one [GenerationEvent.Finished].
     */
    public fun generate(request: GenerationRequest): Flow<GenerationEvent>

    /** One vector per input, in order. Only when [supportsEmbeddings]. */
    public suspend fun embed(inputs: List<String>): List<FloatArray>

    override fun close()
}
