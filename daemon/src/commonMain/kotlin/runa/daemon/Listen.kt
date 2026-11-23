package runa.daemon

import io.ktor.server.application.Application
import io.ktor.server.application.serverConfig
import io.ktor.server.cio.CIO
import io.ktor.server.cio.CIOApplicationEngine
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.connector
import io.ktor.server.engine.embeddedServer
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import runa.core.UserInputError

internal typealias HttpServer = EmbeddedServer<CIOApplicationEngine, CIOApplicationEngine.Configuration>

/**
 * Starts an HTTP server on [host] and [port] (0 for any free port) and returns once it listens.
 *
 * @throws UserInputError when the address cannot be listened on, for example because the port is taken.
 */
internal suspend fun listen(
    host: String,
    port: Int,
    log: DaemonLog,
    module: Application.() -> Unit,
): HttpServer {
    // Ktor reports a failed bind twice: to whoever starts the server, and as an uncaught exception
    // of its accept loop. On native an uncaught exception ends the process, so it is caught here.
    val failures = CoroutineExceptionHandler { _, error -> log.error("The listener on $host:$port failed: ${error.message ?: error}") }
    val server =
        embeddedServer(
            CIO,
            serverConfig {
                parentCoroutineContext = failures
                module(module)
            },
        ) {
            connector {
                this.host = host
                this.port = port
            }
        }
    try {
        server.startSuspend(wait = false)
    } catch (e: Exception) {
        // A failed bind arrives as the cancellation of Ktor's own job; when this coroutine is the
        // one being cancelled, that is not a failure to report.
        if (e is CancellationException) currentCoroutineContext().ensureActive()
        val reason = (e as? CancellationException)?.cause ?: e
        // The application has started even though nothing listens: without this it would stay.
        runCatching { server.stopSuspend(gracePeriodMillis = 0, timeoutMillis = 0) }
        throw UserInputError("Cannot listen on $host:$port: ${reason.message ?: reason::class.simpleName}")
    }
    return server
}

/** The port a server really listens on, which is news when it asked for any free one. */
internal suspend fun HttpServer.port(): Int =
    engine
        .resolvedConnectors()
        .first()
        .port
