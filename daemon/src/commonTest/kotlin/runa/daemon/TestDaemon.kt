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
import runa.client.DaemonClient
import runa.core.LogLevel
import runa.engine.testing.FakeEngine
import runa.store.StoreListing
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

        /** What the `created` fields say. */
        const val NOW = 1_700_000_000L
    }
}

internal fun withDaemon(
    models: StoreListing = listing,
    block: suspend TestDaemon.() -> Unit,
) = runBlocking {
    val engine = FakeEngine()
    val timers = CoroutineScope(SupervisorJob())
    val lines = mutableListOf<String>()
    val log = DaemonLog(LogLevel.INFO) { lines += it.substringAfter(' ') }
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
            server = public,
            version = "1.2.3",
            pid = 42,
            port = { port },
            started = TimeSource.Monotonic.markNow(),
            log = log,
            requestShutdown = { shutdown = true },
        )
    val server = embeddedServer(CIO, port = 0, host = "127.0.0.1") { api.install(this, Listener.PRIVATE) }
    server.startSuspend(wait = false)
    val http = HttpClient(ClientCIO)
    try {
        port =
            server.engine
                .resolvedConnectors()
                .first()
                .port
        TestDaemon("http://127.0.0.1:$port", engine, registry, public, http, lines) { shutdown }.block()
    } finally {
        http.close()
        public.stop()
        server.stopSuspend(gracePeriodMillis = 0, timeoutMillis = 1_000)
        timers.cancel()
    }
}
