package runa.daemon

import io.ktor.client.HttpClient
import io.ktor.client.request.get
import io.ktor.client.statement.bodyAsText
import io.ktor.server.cio.CIO
import io.ktor.server.engine.embeddedServer
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import io.ktor.client.engine.cio.CIO as ClientCIO

// Phase 0 spike (PLAN.md): a Ktor CIO server on native answers a real loopback request.
class CioServerSpikeTest {
    @Test
    fun answersRequest() =
        runBlocking {
            val server =
                embeddedServer(CIO, port = 0, host = "127.0.0.1") {
                    routing { get("/ping") { call.respondText("pong") } }
                }
            server.startSuspend(wait = false)
            try {
                val port =
                    server.engine
                        .resolvedConnectors()
                        .first()
                        .port
                HttpClient(ClientCIO).use { client ->
                    assertEquals("pong", client.get("http://127.0.0.1:$port/ping").bodyAsText())
                }
            } finally {
                server.stopSuspend(gracePeriodMillis = 0, timeoutMillis = 1_000)
            }
        }
}
