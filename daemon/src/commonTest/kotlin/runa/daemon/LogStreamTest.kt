package runa.daemon

import io.ktor.client.request.bearerAuth
import io.ktor.client.request.prepareGet
import io.ktor.client.statement.bodyAsChannel
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.utils.io.readLine
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import runa.core.LogLevel
import runa.core.api.ChatStats
import runa.core.api.DaemonRoutes
import runa.core.api.LogEvent
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

/** `GET /api/v1/logs/stream`, and what the endpoints that generate tell the log, as a client on a loopback socket sees it. */
class LogStreamTest {
    /** The first [expected] events that a listener hears while [work] runs and after it. */
    private suspend fun TestDaemon.listen(
        source: String? = null,
        filter: String? = null,
        expected: Int,
        work: suspend () -> Unit,
    ): List<LogEvent> =
        coroutineScope {
            val listening = CompletableDeferred<Unit>()
            val events = Channel<LogEvent>(Channel.UNLIMITED)
            val listener =
                launch {
                    client().use { client ->
                        client.logs(source, filter, onListening = { listening.complete(Unit) }) { events.trySend(it) }
                    }
                }
            try {
                withTimeout(TIMEOUT) {
                    listening.await()
                    work()
                    List(expected) { events.receive() }
                }
            } finally {
                listener.cancelAndJoin()
            }
        }

    private suspend fun TestDaemon.ask(input: String) = post(DaemonRoutes.CHAT, """{"model": "alpha", "input": "$input"}""")

    @Test
    fun theStreamNeedsTheTokenAndNamesItKnows() =
        withDaemon {
            assertEquals(HttpStatusCode.Unauthorized, get(DaemonRoutes.LOGS, token = null).status)
            val source = get("${DaemonRoutes.LOGS}?source=prompts")
            assertEquals(HttpStatusCode.BadRequest, source.status)
            assertContains(source.bodyAsText(), "source must be one of: model, server, runtime")
            val filter = get("${DaemonRoutes.LOGS}?filter=all")
            assertEquals(HttpStatusCode.BadRequest, filter.status)
            assertContains(filter.bodyAsText(), "filter must be one of: input, output")
        }

    @Test
    fun theStreamSaysWhenItListensAndThenSendsEveryEvent() =
        withDaemon {
            http.prepareGet("$url${DaemonRoutes.LOGS}?source=runtime") { bearerAuth(TestDaemon.TOKEN) }.execute { response ->
                assertEquals(HttpStatusCode.OK, response.status)
                assertEquals(ContentType.Text.EventStream, response.contentType()?.withoutParameters())
                val body = response.bodyAsChannel()
                assertEquals(listOf("event: listening", "data: {}", ""), List(3) { body.readLine() })
                daemonLog.runtime(LogLevel.WARN, "low on memory")
                assertEquals(
                    listOf(
                        "event: log",
                        """data: {"time":"${TestDaemon.TIME}","source":"runtime","type":"log","text":"low on memory","level":"warn"}""",
                        "",
                    ),
                    List(3) { body.readLine() },
                )
            }
        }

    @Test
    fun aChatIsHeardAsWhatTheModelReadAndWhatItWrote() =
        withDaemon {
            val events = listen(LogEvent.MODEL, expected = 2) { ask("Hello there") }
            assertEquals(
                listOf(
                    LogEvent(TestDaemon.TIME, LogEvent.MODEL, LogEvent.INPUT, "user: Hello there", model = ALPHA, endpoint = "chat"),
                    LogEvent(
                        TestDaemon.TIME,
                        LogEvent.MODEL,
                        LogEvent.OUTPUT,
                        "user: Hello there",
                        model = ALPHA,
                        endpoint = "chat",
                        stats = ChatStats(3, 3, 6.67, 0.01, "stop"),
                    ),
                ),
                events,
            )
            // Somebody listened, and still nothing of the conversation was written down.
            assertTrue(log.none { "Hello there" in it }, "$log")
        }

    @Test
    fun theOpenAiEndpointsAreHeardToo() =
        withDaemon {
            val tools = """"tools": [{"type": "function", "function": {"name": "lookup"}}]"""
            val events =
                listen(filter = LogEvent.OUTPUT, expected = 2) {
                    post(
                        OpenAiRoutes.CHAT_COMPLETIONS,
                        """{"model": "alpha", "stream": true, "messages": [{"role": "user", "content": "lookup it"}], $tools}""",
                    ).bodyAsText()
                    post(OpenAiRoutes.COMPLETIONS, """{"model": "alpha", "prompt": "one two"}""")
                }
            assertEquals(listOf("chat.completions", "completions"), events.map { it.endpoint })
            assertEquals(listOf(LogEvent.ToolCall("lookup", """{"words":3}""")), events[0].toolCalls)
            assertEquals("tool_calls", events[0].stats?.stopReason)
            assertEquals("one two", events[1].text)
            assertEquals(null, events[1].toolCalls)
        }

    @Test
    fun aFilterKeepsWhatTheModelsReadOrWhatTheyWrote() =
        withDaemon {
            val written =
                listen(LogEvent.MODEL, LogEvent.OUTPUT, expected = 2) {
                    ask("one")
                    ask("two")
                }
            assertEquals(listOf("output" to "user: one", "output" to "user: two"), written.map { it.type to it.text })
            val read =
                listen(LogEvent.MODEL, LogEvent.INPUT, expected = 2) {
                    ask("three")
                    ask("four")
                }
            assertEquals(listOf("input" to "user: three", "input" to "user: four"), read.map { it.type to it.text })
        }

    @Test
    fun theServerSourceHasTheLogAndEveryRequest() =
        withDaemon {
            val events =
                listen(LogEvent.SERVER, expected = 4) {
                    get(DaemonRoutes.STATUS)
                    get("/nowhere")
                    get(DaemonRoutes.MODELS, token = null)
                    daemonLog.info("Something happened")
                }
            // A request is told when its answer has gone out, which the next one need not wait for.
            assertEquals(
                setOf(
                    "request debug GET /api/v1/status 200 N ms",
                    "request debug GET /nowhere 404 N ms",
                    "request debug GET /api/v1/models 401 N ms",
                    "log info Something happened",
                ),
                events.map { "${it.type} ${it.level} ${it.text.replace(Regex("\\d+ ms$"), "N ms")}" }.toSet(),
            )
            // Clients ask for the status all the time: the file has the requests only at the debug level.
            assertEquals(listOf("INFO  Something happened"), log)
        }

    @Test
    fun aSourceIsHeardWithoutTheOthers() =
        withDaemon {
            val events =
                listen(LogEvent.RUNTIME, expected = 1) {
                    daemonLog.info("from the daemon")
                    ask("from a model")
                    daemonLog.runtime(LogLevel.DEBUG, "from the engine")
                }
            assertEquals(listOf(LogEvent(TestDaemon.TIME, LogEvent.RUNTIME, LogEvent.LOG, "from the engine", level = "debug")), events)
        }

    @Test
    fun theConversationIsWrittenDownOnlyWhenTheDaemonWasToldTo() {
        withDaemon {
            ask("Hello there")
            assertTrue(log.none { "Hello there" in it }, "$log")
        }
        withDaemon(logPrompts = true) {
            ask("Hello there")
            assertContains(log, """INFO  chat '$ALPHA' input: "user: Hello there"""")
            assertContains(log, """INFO  chat '$ALPHA' output: "user: Hello there"""")
        }
    }

    private companion object {
        val TIMEOUT = 10.seconds
    }
}
