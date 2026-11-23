package runa.engine.llama

import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.StableRef
import kotlinx.cinterop.asStableRef
import kotlinx.cinterop.copy
import kotlinx.cinterop.staticCFunction
import kotlinx.cinterop.toKString
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.IO
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.job
import kotlinx.coroutines.withContext
import okio.FileSystem
import okio.IOException
import okio.Path
import runa.core.LoadConfig
import runa.core.LogLevel
import runa.core.UserInputError
import runa.engine.Engine
import runa.engine.EngineInfo
import runa.engine.LoadedModel
import runa.engine.MemoryEstimate
import runa.engine.llama.cinterop.ggml_backend_reg_count
import runa.engine.llama.cinterop.ggml_backend_reg_get
import runa.engine.llama.cinterop.ggml_backend_reg_name
import runa.engine.llama.cinterop.llama_context_default_params
import runa.engine.llama.cinterop.llama_init_from_model
import runa.engine.llama.cinterop.llama_model_default_params
import runa.engine.llama.cinterop.llama_model_free
import runa.engine.llama.cinterop.llama_model_load_from_file
import runa.engine.llama.cinterop.llama_supports_gpu_offload
import runa.gguf.GgufFormatException
import runa.gguf.GgufHeader
import runa.gguf.GgufSplitPart
import runa.gguf.readGgufHeader
import runa.platform.computeCoreCount
import runa.platform.resolvedMetadataOrNull
import kotlin.coroutines.coroutineContext

/**
 * The llama.cpp [Engine]. [onLog] receives llama.cpp's own log at [LogLevel.DEBUG] and the engine's
 * notices at their own level, a line at a time and from any thread; llama.cpp has a single log per
 * process, so the engine created last gets it.
 */
public class LlamaEngine(
    private val onLog: (LogLevel, String) -> Unit = { _, _ -> },
) : Engine {
    init {
        startLlama(onLog)
    }

    override val info: EngineInfo by lazy {
        val backends =
            (0 until ggml_backend_reg_count().toInt())
                .mapNotNull { ggml_backend_reg_name(ggml_backend_reg_get(it.toULong()))?.toKString()?.lowercase() }
        EngineInfo(
            name = "llama.cpp",
            version = LlamaBuild.VERSION,
            accelerators = listOf(CPU) + (backends - CPU),
            gpuOffload = llama_supports_gpu_offload(),
        )
    }

    override fun estimateMemory(
        model: Path,
        config: LoadConfig,
    ): MemoryEstimate {
        val header = readHeader(model)
        return MemoryEstimate(weightsBytes = fileBytes(model), contextBytes = kvCacheBytes(header, contextLength(header, config).toLong()))
    }

    override suspend fun load(
        model: Path,
        config: LoadConfig,
        onProgress: (Float) -> Unit,
    ): LoadedModel = load(model, config, pooling = null, onProgress)

    /** [pooling] turns any model into an embedding model with that `llama_pooling_type`; only tests use it. */
    internal suspend fun load(
        model: Path,
        config: LoadConfig,
        pooling: Int?,
        onProgress: (Float) -> Unit = {},
    ): LoadedModel {
        val header = readHeader(model)
        val type = header.type ?: GgufHeader.TYPE_MODEL
        if (type != GgufHeader.TYPE_MODEL) throw UserInputError("$model holds a GGUF $type, not a model")

        // llama.cpp is not thread-safe, so each model gets a dispatcher that runs one call at a time.
        val serial = Dispatchers.IO.limitedParallelism(1)
        var opened: LlamaModel? = null
        try {
            withContext(serial) { opened = open(model, header, config, pooling, onProgress, serial, coroutineContext.job) }
        } catch (e: Throwable) {
            // Cancelled between the load and the return: nobody else will ever close it.
            opened?.close()
            throw e
        }
        return checkNotNull(opened)
    }

    private fun open(
        path: Path,
        header: GgufHeader,
        config: LoadConfig,
        pooling: Int?,
        onProgress: (Float) -> Unit,
        serial: CoroutineDispatcher,
        job: Job,
    ): LlamaModel {
        val progress = LoadProgress(job, onProgress)
        val reference = StableRef.create(progress)
        takeLastLlamaError()
        val model =
            try {
                val parameters =
                    llama_model_default_params().copy {
                        n_gpu_layers = gpuLayerCount(config.gpu, header.architectureLong("block_count"))
                        progress_callback = staticCFunction(::onLoadProgress)
                        progress_callback_user_data = reference.asCPointer()
                    }
                llama_model_load_from_file(path.toString(), parameters)
            } finally {
                reference.dispose()
            }
        progress.failure?.let { failure ->
            model?.let { llama_model_free(it) }
            throw failure
        }
        if (model == null) {
            job.ensureActive()
            throw UserInputError("llama.cpp cannot load $path${reason()}")
        }

        val embeddings = pooling != null || header.isEmbeddingModel
        val length = contextLength(header, config)
        val threads = config.threads ?: computeCoreCount()
        val parameters =
            llama_context_default_params().copy {
                n_ctx = length.toUInt()
                n_threads = threads
                n_threads_batch = threads
                this.embeddings = embeddings
                if (pooling != null) pooling_type = pooling
                // A pooled embedding needs its whole text in one batch.
                if (embeddings && length > 0) {
                    n_batch = length.toUInt()
                    n_ubatch = length.toUInt()
                }
            }
        val context = llama_init_from_model(model, parameters)
        if (context == null) {
            llama_model_free(model)
            throw UserInputError("llama.cpp cannot create a context of $length tokens for $path${reason()}")
        }
        return LlamaModel(model, context, embeddings, serial, onLog)
    }

    private fun readHeader(model: Path): GgufHeader =
        try {
            readGgufHeader(FileSystem.SYSTEM, model)
        } catch (e: GgufFormatException) {
            throw UserInputError("$model is not a GGUF model: ${e.message}")
        } catch (e: IOException) {
            throw UserInputError("Cannot read the model file $model: ${e.message}")
        }

    /** The size of the model on disk; for a split model, of all its parts. */
    private fun fileBytes(model: Path): Long {
        val part = GgufSplitPart.parse(model.name)
        val directory = model.parent
        val files = if (part == null || directory == null) listOf(model) else (1..part.total).map { directory / part.fileName(it) }
        return files.sumOf { FileSystem.SYSTEM.resolvedMetadataOrNull(it)?.size ?: 0L }
    }

    private fun reason(): String = takeLastLlamaError()?.let { ": ${it.trim()}" }.orEmpty()

    private companion object {
        const val CPU = "cpu"

        /**
         * The context to load with: what was asked for, but no more than the model was trained
         * with. 0 leaves it to llama.cpp when the file does not say.
         */
        fun contextLength(
            header: GgufHeader,
            config: LoadConfig,
        ): Int {
            val trained = header.contextLength?.coerceAtMost(Int.MAX_VALUE.toLong())?.toInt()
            val wanted = config.contextLength
            return when {
                wanted == null -> trained ?: 0
                trained == null -> wanted
                else -> minOf(wanted, trained)
            }
        }
    }
}

private class LoadProgress(
    private val job: Job,
    private val onProgress: (Float) -> Unit,
) {
    var failure: Throwable? = null
        private set

    /** False tells llama.cpp to stop loading. */
    fun report(progress: Float): Boolean {
        if (!job.isActive) return false
        return try {
            onProgress(progress)
            true
        } catch (e: Throwable) {
            // Kept for the caller: an exception cannot cross llama.cpp's C++ frames.
            failure = e
            false
        }
    }
}

private fun onLoadProgress(
    progress: Float,
    data: COpaquePointer?,
): Boolean = data?.asStableRef<LoadProgress>()?.get()?.report(progress) ?: true
