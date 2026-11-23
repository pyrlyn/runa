package runa.engine.llama

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.staticCFunction
import kotlinx.cinterop.toKString
import runa.core.LogLevel
import runa.engine.llama.cinterop.GGML_LOG_LEVEL_CONT
import runa.engine.llama.cinterop.GGML_LOG_LEVEL_ERROR
import runa.engine.llama.cinterop.llama_backend_init
import runa.engine.llama.cinterop.llama_log_set
import kotlin.concurrent.Volatile
import kotlin.concurrent.atomics.AtomicReference
import kotlin.concurrent.atomics.ExperimentalAtomicApi

// llama.cpp has one log callback per process and calls it from its own threads with pieces of
// lines, so the state lives here rather than in an engine instance.
@OptIn(ExperimentalAtomicApi::class)
private val sink = AtomicReference<(LogLevel, String) -> Unit> { _, _ -> }

@OptIn(ExperimentalAtomicApi::class)
private val openLine = AtomicReference("")

@OptIn(ExperimentalAtomicApi::class)
private val lastError = AtomicReference<String?>(null)

@Volatile
private var lineIsError = false

private val backend: Unit by lazy {
    llama_log_set(staticCFunction(::onLog), null)
    llama_backend_init()
}

/**
 * Initialises llama.cpp once and sends its log to [onLog] from now on, one whole line per call and
 * always at [LogLevel.DEBUG]: llama.cpp narrates every load at "info" and "warn", which is noise to
 * someone who asked a model a question. What matters of it reaches the user through
 * [takeLastLlamaError].
 */
@OptIn(ExperimentalAtomicApi::class)
internal fun startLlama(onLog: (LogLevel, String) -> Unit) {
    sink.store(onLog)
    backend
}

/** The last error llama.cpp logged since the previous call: the only place it says why something failed. */
@OptIn(ExperimentalAtomicApi::class)
internal fun takeLastLlamaError(): String? = lastError.exchange(null)

@OptIn(ExperimentalAtomicApi::class)
private fun onLog(
    level: UInt,
    text: CPointer<ByteVar>?,
    @Suppress("UNUSED_PARAMETER") data: COpaquePointer?,
) {
    val piece = text?.toKString() ?: return
    if (level != GGML_LOG_LEVEL_CONT) lineIsError = level == GGML_LOG_LEVEL_ERROR
    while (true) {
        val before = openLine.load()
        val all = before + piece
        val end = all.lastIndexOf('\n')
        if (!openLine.compareAndSet(before, all.substring(end + 1))) continue
        if (end >= 0) {
            all
                .substring(0, end)
                .lineSequence()
                .filter { it.isNotBlank() }
                .forEach(::report)
        }
        return
    }
}

@OptIn(ExperimentalAtomicApi::class)
private fun report(line: String) {
    if (lineIsError) lastError.store(line)
    try {
        sink.load()(LogLevel.DEBUG, line)
    } catch (_: Throwable) {
        // An exception must not unwind into llama.cpp's C++ frames; a lost log line is harmless.
    }
}
