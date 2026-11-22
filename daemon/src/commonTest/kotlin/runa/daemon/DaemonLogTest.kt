package runa.daemon

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import runa.core.LogLevel
import runa.core.ToolCall
import runa.core.api.LogEvent
import runa.engine.Completion
import runa.engine.FinishReason
import runa.engine.GenerationStats
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Instant

@OptIn(ExperimentalCoroutinesApi::class)
class DaemonLogTest {
    private val lines = mutableListOf<String>()

    private fun log(
        threshold: LogLevel = LogLevel.INFO,
        logPrompts: Boolean = false,
    ) = DaemonLog(threshold, logPrompts, now = { Instant.fromEpochSeconds(0) }) { lines += it }

    private val completion =
        Completion(
            "Let me\nlook.",
            FinishReason.TOOL_CALLS,
            GenerationStats(7, 3, 250.milliseconds, 240.milliseconds),
            listOf(ToolCall("call_0", "get_weather", """{"city":"Oslo"}""")),
        )

    @Test
    fun aLineBelowTheThresholdIsHeardButNotWritten() =
        runTest {
            val log = log(LogLevel.WARN)
            val heard = mutableListOf<LogEvent>()
            backgroundScope.launch { log.events.toList(heard) }
            runCurrent()

            log.info("quiet")
            log.error("loud")
            log.runtime(LogLevel.WARN, "from the engine")
            log.request("GET /api/v1/status 200 1 ms")
            runCurrent()

            assertEquals(listOf("$EPOCH ERROR loud", "$EPOCH WARN  from the engine"), lines)
            assertEquals(
                listOf(
                    "server log info quiet",
                    "server log error loud",
                    "runtime log warn from the engine",
                    "server request debug GET /api/v1/status 200 1 ms",
                ),
                heard.map { "${it.source} ${it.type} ${it.level} ${it.text}" },
            )
        }

    @Test
    fun aConversationIsWorthPuttingTogetherOnlyForAListenerOrTheFile() =
        runTest {
            val log = log()
            assertFalse(log.wantsModelEvents)
            val listener = backgroundScope.launch { log.events.collect {} }
            runCurrent()
            assertTrue(log.wantsModelEvents)
            listener.cancel()
            runCurrent()
            assertFalse(log.wantsModelEvents)
            assertTrue(log(logPrompts = true).wantsModelEvents)
        }

    @Test
    fun aConversationIsNotWrittenUnlessAsked() {
        val log = log(LogLevel.DEBUG)
        log.input("chat", "alpha", "user: Hi")
        log.output("chat", "alpha", completion, chatStats(completion.stats, completion.reason))
        assertEquals(emptyList(), lines)
    }

    @Test
    fun aWrittenConversationKeepsEveryTextOnOneLine() {
        // Even a daemon that logs errors only: it was asked for the conversations.
        val log = log(LogLevel.ERROR, logPrompts = true)
        log.input("chat.completions", "alpha", "system: Be brief\nuser: Hi")
        log.output("chat.completions", "alpha", completion, chatStats(completion.stats, completion.reason))
        assertEquals(
            listOf(
                """$EPOCH INFO  chat.completions 'alpha' input: "system: Be brief\nuser: Hi"""",
                """$EPOCH INFO  chat.completions 'alpha' output: "Let me\nlook." get_weather({"city":"Oslo"})""",
            ),
            lines,
        )
    }

    private companion object {
        const val EPOCH = "1970-01-01T00:00:00Z"
    }
}
