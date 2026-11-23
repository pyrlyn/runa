package runa.daemon

import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.serialization.json.JsonPrimitive
import runa.core.LogLevel
import runa.core.api.ChatStats
import runa.core.api.LogEvent
import runa.engine.Completion
import kotlin.time.Clock
import kotlin.time.Instant

/**
 * The daemon's log. Everything that happens is an event that `runa log stream` reads while it
 * happens; a line at or above [threshold] is also written to stdout, which the process that started
 * the daemon pointed at `logs/daemon.log`.
 *
 * What a model read and wrote is somebody's conversation. It reaches whoever listens at that
 * moment and is written down only when [logPrompts] says so.
 */
internal class DaemonLog(
    private val threshold: LogLevel,
    private val logPrompts: Boolean = false,
    private val now: () -> Instant = Clock.System::now,
    private val write: (String) -> Unit = ::println,
) {
    // Nobody who listens may hold up the daemon, least of all llama.cpp's own threads: a listener
    // that is too slow loses the oldest events.
    private val bus = MutableSharedFlow<LogEvent>(extraBufferCapacity = BUFFERED_EVENTS, onBufferOverflow = BufferOverflow.DROP_OLDEST)

    /** Every event from now on. Nothing is kept for a listener that comes later. */
    val events: SharedFlow<LogEvent> = bus.asSharedFlow()

    /** True when what a model reads and writes has somewhere to go, so that it is worth putting together. */
    val wantsModelEvents: Boolean get() = logPrompts || bus.subscriptionCount.value > 0

    fun info(message: String) = at(LogLevel.INFO, message)

    fun error(message: String) = at(LogLevel.ERROR, message)

    fun at(
        level: LogLevel,
        message: String,
    ) = line(LogEvent.SERVER, LogEvent.LOG, level, message)

    /** A line of the inference engine's own log. May be called from any thread. */
    fun runtime(
        level: LogLevel,
        message: String,
    ) = line(LogEvent.RUNTIME, LogEvent.LOG, level, message)

    /** A request that was answered. Clients ask for the status all the time, so the file has these only when it has everything. */
    fun request(description: String) = line(LogEvent.SERVER, LogEvent.REQUEST, LogLevel.DEBUG, description)

    /** What the instance [model] reads for a request to [endpoint]. */
    fun input(
        endpoint: String,
        model: String,
        prompt: String,
    ) = conversation(LogEvent(time(), LogEvent.MODEL, LogEvent.INPUT, prompt, model = model, endpoint = endpoint))

    /** What the instance [model] wrote for a request to [endpoint]. */
    fun output(
        endpoint: String,
        model: String,
        completion: Completion,
        stats: ChatStats,
    ) = conversation(
        LogEvent(
            time = time(),
            source = LogEvent.MODEL,
            type = LogEvent.OUTPUT,
            text = completion.text,
            model = model,
            endpoint = endpoint,
            toolCalls = completion.toolCalls.map { LogEvent.ToolCall(it.name, it.arguments) }.ifEmpty { null },
            stats = stats,
        ),
    )

    private fun line(
        source: String,
        type: String,
        level: LogLevel,
        message: String,
    ) {
        val time = time()
        bus.tryEmit(LogEvent(time, source, type, message, level = level.id))
        if (level >= threshold) write("$time ${level.id.uppercase().padEnd(LEVEL_WIDTH)} $message")
    }

    private fun conversation(event: LogEvent) {
        bus.tryEmit(event)
        if (!logPrompts) return
        // As JSON, so that a text of many lines stays one line of the log.
        val calls = event.toolCalls.orEmpty().joinToString("") { " ${it.name}(${it.arguments})" }
        write(
            "${event.time} ${LogLevel.INFO.id.uppercase().padEnd(LEVEL_WIDTH)} " +
                "${event.endpoint} '${event.model}' ${event.type}: ${JsonPrimitive(event.text)}$calls",
        )
    }

    private fun time() = now().toString()

    private companion object {
        const val LEVEL_WIDTH = 5

        /** llama.cpp writes a few hundred lines when it loads a model; a listener may fall this far behind. */
        const val BUFFERED_EVENTS = 1024
    }
}
