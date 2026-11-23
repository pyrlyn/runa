package runa.daemon

import io.ktor.http.HttpHeaders
import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.plugins.cors.CORSConfig
import io.ktor.server.plugins.cors.routing.CORS
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import runa.core.LogLevel
import runa.core.UserInputError
import runa.core.api.ServerDescription
import runa.core.api.ServerRequest
import runa.core.isLoopbackHost

/**
 * The listener of `runa server start`: the daemon's API on an address and port that the user
 * chose, for programs that cannot read the discovery file. The daemon's own loopback port is not
 * affected by it, so the CLI keeps working whether this one is open or not.
 *
 * @param afterStart run once the listener is open; Ktor installs signal handlers of its own then.
 */
internal class PublicServer(
    private val access: Access,
    private val log: DaemonLog,
    private val afterStart: () -> Unit = {},
) {
    private val lock = Mutex()
    private var open: Open? = null

    suspend fun describe(): ServerDescription = lock.withLock { open?.description ?: ServerDescription() }

    /**
     * Opens the listener with [routes]. Asking again for what is open already changes nothing.
     *
     * @throws UserInputError when it is open with other settings, or the address cannot be listened on.
     */
    suspend fun start(
        request: ServerRequest,
        routes: Application.() -> Unit,
    ): ServerDescription =
        lock.withLock {
            val bind = request.bind?.takeIf { it.isNotBlank() } ?: throw UserInputError("bind is required")
            val port = request.port?.takeIf { it in PORTS } ?: throw UserInputError("port must be a number from 1 to 65535")
            val apiKey = request.apiKey
            if (apiKey != null && apiKey.isBlank()) throw UserInputError("api_key must not be empty")
            val description =
                ServerDescription(
                    running = true,
                    bind = bind,
                    port = port,
                    cors = request.cors,
                    apiKeyRequired =
                        apiKey != null,
                )
            open?.let {
                if (it.description == description && it.apiKey == apiKey) return@withLock description
                throw UserInputError(
                    "The server already runs on ${it.description.bind}:${it.description.port}; stop it first with `runa server stop`",
                )
            }
            val server =
                try {
                    listen(bind, port, log) {
                        if (request.cors) install(CORS) { allowBrowsers() }
                        routes()
                    }
                } finally {
                    afterStart()
                }
            access.apiKey = apiKey
            open = Open(server, description, apiKey)
            log.info(
                "The server listens on $bind:$port" + (if (request.cors) ", with CORS" else "") +
                    (if (apiKey != null) ", with an API key" else ""),
            )
            if (apiKey == null && !isLoopbackHost(bind)) {
                log.at(LogLevel.WARN, "The server can be reached from other machines and asks for no API key")
            }
            description
        }

    /** Closes the listener; returns false when it was not open. */
    suspend fun stop(): Boolean =
        lock.withLock {
            val current = open ?: return@withLock false
            open = null
            access.apiKey = null
            current.server.stopSuspend(gracePeriodMillis = GRACE_MILLIS, timeoutMillis = STOP_TIMEOUT_MILLIS)
            log.info("The server on ${current.description.bind}:${current.description.port} is closed")
            true
        }

    private class Open(
        val server: HttpServer,
        val description: ServerDescription,
        val apiKey: String?,
    )

    private companion object {
        val PORTS = 1..65535
        const val GRACE_MILLIS = 200L
        const val STOP_TIMEOUT_MILLIS = 2_000L

        /** What a page in a browser needs to call the API: any origin, JSON bodies and the key header. */
        fun CORSConfig.allowBrowsers() {
            anyHost()
            allowHeader(HttpHeaders.Authorization)
            allowHeader(HttpHeaders.ContentType)
            allowNonSimpleContentTypes = true
        }
    }
}
