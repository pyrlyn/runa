package runa.daemon

import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.options
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.core.api.ServerDescription
import runa.core.api.ServerRequest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The listener of `runa server start`, beside the daemon's own. */
class PublicServerTest {
    private suspend fun TestDaemon.open(
        port: Int,
        cors: Boolean = false,
        apiKey: String? = null,
    ): ServerDescription =
        client().use { it.server(ServerRequest(running = true, bind = LOOPBACK, port = port, cors = cors, apiKey = apiKey)) }

    private suspend fun TestDaemon.close(): ServerDescription = client().use { it.server(ServerRequest(running = false)) }

    private suspend fun assertUnauthorized(response: HttpResponse) {
        assertEquals(HttpStatusCode.Unauthorized, response.status, response.bodyAsText())
    }

    @Test
    fun theSameApiOnASecondAddress() =
        withDaemon {
            val port = freePort()
            val public = "http://$LOOPBACK:$port"
            assertEquals(ServerDescription(), client().use { it.status().server })

            val description = open(port)
            assertEquals(ServerDescription(running = true, bind = LOOPBACK, port = port), description)
            assertEquals(description, client().use { it.status().server })
            assertTrue(log.any { it.endsWith("The server listens on $LOOPBACK:$port") }, "$log")

            // Programs that cannot know the token can still use a model.
            assertEquals(HttpStatusCode.OK, get(OpenAiRoutes.MODELS, token = null, base = public).status)
            val answer =
                post(OpenAiRoutes.CHAT_COMPLETIONS, """{"model": "alpha", "messages": [{"role": "user", "content": "Hi"}]}""", null, public)
            assertEquals(HttpStatusCode.OK, answer.status)
            // The daemon itself stays behind its token, the status too.
            assertUnauthorized(get(DaemonRoutes.MODELS, token = null, base = public))
            assertUnauthorized(get(DaemonRoutes.STATUS, token = null, base = public))
            assertUnauthorized(post(DaemonRoutes.SHUTDOWN, "", token = null, base = public))
            assertEquals(HttpStatusCode.OK, get(DaemonRoutes.STATUS, base = public).status)
            // And its own port is what it was.
            assertUnauthorized(get(OpenAiRoutes.MODELS, token = null))
            assertEquals(HttpStatusCode.OK, get(DaemonRoutes.STATUS, token = null).status)

            assertEquals(ServerDescription(), close())
            assertFailsWith<Exception> { get(OpenAiRoutes.MODELS, token = null, base = public) }
            assertEquals(ServerDescription(), close())
            assertEquals(1, engine.loads.size)
        }

    @Test
    fun askingAgainChangesNothingAndOtherSettingsNeedAStop() =
        withDaemon {
            val port = freePort()
            val first = open(port)
            assertEquals(first, open(port))
            val other = freePort()
            for (attempt in listOf<suspend () -> Unit>({ open(other) }, { open(port, cors = true) }, { open(port, apiKey = "secret") })) {
                val error = assertFailsWith<UserInputError> { attempt() }
                assertTrue("already runs on $LOOPBACK:$port" in error.message.orEmpty(), error.message)
            }
            assertEquals(first, client().use { it.status().server })
            close()
            assertEquals(other, open(other).port)
        }

    @Test
    fun anApiKeyClosesTheOpenRoutes() =
        withDaemon {
            val port = freePort()
            val public = "http://$LOOPBACK:$port"
            assertTrue(open(port, apiKey = "sk-local").apiKeyRequired)
            assertTrue(log.any { it.endsWith("with an API key") }, "$log")

            for (token in listOf(null, "", "sk-loca", "sk-local2")) assertUnauthorized(get(OpenAiRoutes.MODELS, token, public))
            assertEquals(HttpStatusCode.OK, get(OpenAiRoutes.MODELS, "sk-local", public).status)
            assertEquals(HttpStatusCode.OK, get(OpenAiRoutes.MODELS, TestDaemon.TOKEN, public).status)
            // The key is for using models, not for running the daemon; and it means nothing on the daemon's own port.
            assertUnauthorized(get(DaemonRoutes.MODELS, "sk-local", public))
            assertUnauthorized(post(DaemonRoutes.SHUTDOWN, "", "sk-local", public))
            assertUnauthorized(get(OpenAiRoutes.MODELS, "sk-local"))
            assertFalse(shutdownRequested())

            close()
            assertFalse(open(port).apiKeyRequired)
            assertEquals(HttpStatusCode.OK, get(OpenAiRoutes.MODELS, token = null, base = public).status)
        }

    @Test
    fun corsAnswersABrowsersQuestion() =
        withDaemon {
            val port = freePort()
            val public = "http://$LOOPBACK:$port"

            suspend fun preflight() =
                http.options(public + OpenAiRoutes.CHAT_COMPLETIONS) {
                    header(HttpHeaders.Origin, "https://app.example")
                    header(HttpHeaders.AccessControlRequestMethod, "POST")
                    header(HttpHeaders.AccessControlRequestHeaders, "authorization,content-type")
                }

            suspend fun models() = http.get(public + OpenAiRoutes.MODELS) { header(HttpHeaders.Origin, "https://app.example") }

            open(port)
            assertNull(models().headers[HttpHeaders.AccessControlAllowOrigin])
            close()

            assertTrue(open(port, cors = true).cors)
            val allowed = preflight()
            assertEquals(HttpStatusCode.OK, allowed.status)
            assertEquals("*", allowed.headers[HttpHeaders.AccessControlAllowOrigin])
            val headers = allowed.headers[HttpHeaders.AccessControlAllowHeaders].orEmpty().lowercase()
            assertTrue("authorization" in headers && "content-type" in headers, headers)
            assertEquals("*", models().headers[HttpHeaders.AccessControlAllowOrigin])
        }

    @Test
    fun whatCannotBeListenedOnIsTheUsersMistake() =
        withDaemon {
            val taken = url.substringAfterLast(':').toInt()
            val error = assertFailsWith<UserInputError> { open(taken) }
            assertTrue("Cannot listen on $LOOPBACK:$taken" in error.message.orEmpty(), error.message)
            assertEquals(ServerDescription(), client().use { it.status().server })
            // The daemon's own listener is unharmed.
            assertEquals(HttpStatusCode.OK, get(DaemonRoutes.MODELS).status)

            suspend fun refused(body: String): String {
                val response = post(DaemonRoutes.SERVER, body)
                assertEquals(HttpStatusCode.BadRequest, response.status)
                return DaemonJson.decodeFromString(ApiError.serializer(), response.bodyAsText()).error.message
            }
            assertTrue("bind" in refused("""{"running": true, "port": 8000}"""))
            assertTrue("port" in refused("""{"running": true, "bind": "127.0.0.1"}"""))
            assertTrue("port" in refused("""{"running": true, "bind": "127.0.0.1", "port": 70000}"""))
            assertTrue("api_key" in refused("""{"running": true, "bind": "127.0.0.1", "port": 8000, "api_key": " "}"""))
        }

    @Test
    fun itCannotBeClosedThroughItself() =
        withDaemon {
            val port = freePort()
            open(port)
            val response = post(DaemonRoutes.SERVER, """{"running": false}""", base = "http://$LOOPBACK:$port")
            assertEquals(HttpStatusCode.BadRequest, response.status)
            assertTrue(client().use { it.status().server.running })
        }
}
