package runa.daemon

import io.ktor.client.HttpClient
import io.ktor.client.request.bearerAuth
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.server.cio.CIO
import io.ktor.server.engine.embeddedServer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import okio.Path.Companion.toPath
import runa.client.DaemonClient
import runa.core.LogLevel
import runa.engine.testing.FakeEngine
import runa.store.StoreListing
import kotlin.time.Instant
import kotlin.time.TimeSource
import io.ktor.client.engine.cio.CIO as ClientCIO

/** The daemon's API on a real loopback socket, with the fake engine behind it. */
internal class TestDaemon(
    val url: String,
    val engine: FakeEngine,
    val registry: ModelRegistry,
    val server: PublicServer,
    val http: HttpClient,
    /** What the daemon logged, without the timestamps. */
    val log: List<String>,
    val daemonLog: DaemonLog,
    val shutdownRequested: () -> Boolean,
) {
    fun client(token: String? = TOKEN) = DaemonClient(ClientCIO.create(), url, token)

    suspend fun get(
        path: String,
        token: String? = TOKEN,
        base: String = url,
    ): HttpResponse = http.get(base + path) { token?.let { bearerAuth(it) } }

    suspend fun post(
        path: String,
        body: String,
        token: String? = TOKEN,
        base: String = url,
    ): HttpResponse =
        http.post(base + path) {
            token?.let { bearerAuth(it) }
            setBody(body)
        }

    companion object {
        const val TOKEN = "0123456789abcdef"

        /** What the `created` fields say, and the time of every line of the log: [TIME]. */
        const val NOW = 1_700_000_000L
        const val TIME = "2023-11-14T22:13:20Z"

        const val LOG_FILE = "/state/logs/daemon.log"

        /** How many conversations this daemon remembers: few, so a test can make it forget one. */
        const val CONVERSATIONS = 3
    }
}

internal const val LOOPBACK = "127.0.0.1"

/** A port that was free a moment ago. */
internal suspend fun freePort(): Int {
    val probe = embeddedServer(CIO, port = 0, host = LOOPBACK) {}
    probe.startSuspend(wait = false)
    try {
        return probe.engine
            .resolvedConnectors()
            .first()
            .port
    } finally {
        probe.stopSuspend(gracePeriodMillis = 0, timeoutMillis = 1_000)
    }
}

internal fun withDaemon(
    models: StoreListing = listing,
    /** Whether this daemon writes what its models read and write into its log. */
    logPrompts: Boolean = false,
    block: suspend TestDaemon.() -> Unit,
) = runBlocking {
    val engine = FakeEngine()
    val timers = CoroutineScope(SupervisorJob())
    val lines = mutableListOf<String>()
    val log = DaemonLog(LogLevel.INFO, logPrompts, now = { Instant.fromEpochSeconds(TestDaemon.NOW) }) { lines += it.substringAfter(' ') }
    val registry = ModelRegistry(engine, { models }, modelsDir, timers)
    val access = Access(TestDaemon.TOKEN)
    val failures = Failures(log)
    val public = PublicServer(access, log)
    var port = 0
    var shutdown = false
    var ids = 0
    val api =
        DaemonApi(
            registry = registry,
            engine = engine,
            store = { models },
            access = access,
            failures = failures,
            openAi = OpenAiApi(registry, { models }, access, failures, log, now = { TestDaemon.NOW }, newId = { "${++ids}" }),
            chat = ChatApi(registry, failures, log, Conversations(TestDaemon.CONVERSATIONS), newId = { "${++ids}" }),
            server = public,
            version = "1.2.3",
            pid = 42,
            port = { port },
            started = TimeSource.Monotonic.markNow(),
            log = log,
            logFile = TestDaemon.LOG_FILE.toPath(),
            requestShutdown = { shutdown = true },
        )
    val server = embeddedServer(CIO, port = 0, host = LOOPBACK) { api.install(this, Listener.PRIVATE) }
    server.startSuspend(wait = false)
    val http = HttpClient(ClientCIO)
    try {
        port =
            server.engine
                .resolvedConnectors()
                .first()
                .port
        TestDaemon("http://$LOOPBACK:$port", engine, registry, public, http, lines, log) { shutdown }.block()
    } finally {
        http.close()
        public.stop()
        server.stopSuspend(gracePeriodMillis = 0, timeoutMillis = 1_000)
        timers.cancel()
    }
}
