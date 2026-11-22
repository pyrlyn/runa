package runa.daemon

import io.ktor.client.HttpClient
import io.ktor.client.request.bearerAuth
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpStatusCode
import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.UnloadRequest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import io.ktor.client.engine.cio.CIO as ClientCIO

/** The real server and the real client, talking over a loopback socket. */
class DaemonApiTest {
    @Test
    fun statusNeedsNoToken() =
        withDaemon {
            client(token = null).use { client ->
                val status = client.status()
                assertEquals("1.2.3", status.version)
                assertEquals(42, status.pid)
                assertEquals(url.substringAfterLast(':').toInt(), status.port)
                assertEquals("fake", status.engine.name)
                assertEquals(0, status.loadedInstances)
            }
        }

    @Test
    fun everythingElseNeedsTheToken() =
        withDaemon {
            for (token in listOf(null, "", "wrong", TOKEN + "x", TOKEN.dropLast(1))) {
                client(token).use { client ->
                    assertFailsWith<BackendError> { client.models() }
                    assertFailsWith<BackendError> { client.load(LoadRequest("alpha")) }
                    assertFailsWith<BackendError> { client.unload(UnloadRequest(all = true)) }
                    assertFailsWith<BackendError> { client.shutdown() }
                }
            }
            assertEquals(0, engine.loads.size)
            assertEquals(false, shutdownRequested())
        }

    @Test
    fun loadListAndUnload() =
        withDaemon {
            client().use { client ->
                val progress = mutableListOf<Float>()
                val loaded =
                    client.load(LoadRequest("alpha", contextLength = 512, parallel = 2, ttl = 60, identifier = "mine")) {
                        progress +=
                            it
                    }
                assertEquals(LoadResponse.LOADED, loaded.status)
                val instance = assertNotNull(loaded.instance)
                assertEquals("mine", instance.id)
                assertEquals(ALPHA, instance.model)
                assertEquals(512, instance.contextLength)
                assertEquals(2, instance.parallel)
                assertEquals(60, instance.ttlSeconds)
                assertEquals(1_000 + 2 * 512, loaded.estimate.totalBytes)
                assertNotNull(loaded.loadTimeSeconds)
                assertEquals(1f, progress.last())

                assertEquals(1, client.status().loadedInstances)
                val models = client.models().models
                assertEquals(listOf(ALPHA, "acme/beta@Q4_K_M"), models.map { it.key })
                assertEquals(listOf("mine"), models[0].loadedInstances.map { it.id })
                assertEquals(emptyList(), models[1].loadedInstances)
                assertEquals("Tiny alpha", models[0].displayName)

                assertEquals(listOf("mine"), client.unload(UnloadRequest(instanceId = "mine")).unloaded)
                assertTrue(
                    engine.loads
                        .single()
                        .model.closed,
                )
                assertEquals(0, client.status().loadedInstances)
            }
        }

    @Test
    fun aWrongRequestIsTheUsersMistake() =
        withDaemon {
            client().use { client ->
                assertTrue("gamma" in assertFailsWith<UserInputError> { client.load(LoadRequest("gamma")) }.message.orEmpty())
                assertFailsWith<UserInputError> { client.load(LoadRequest("alpha", gpu = "lots")) }
                assertFailsWith<UserInputError> { client.load(LoadRequest("alpha", ttl = -1)) }
                assertFailsWith<UserInputError> { client.load(LoadRequest("alpha", estimateOnly = true, contextLength = 0)) }
                client.load(LoadRequest("alpha", identifier = "mine"))
                assertTrue(
                    "mine" in assertFailsWith<UserInputError> { client.load(LoadRequest("beta", identifier = "mine")) }.message.orEmpty(),
                )
                assertFailsWith<UserInputError> { client.unload(UnloadRequest(instanceId = "other")) }
                assertFailsWith<UserInputError> { client.unload(UnloadRequest()) }
                assertEquals(1, engine.loads.size)
            }
        }

    @Test
    fun anEstimateLoadsNothing() =
        withDaemon {
            client().use { client ->
                val response = client.load(LoadRequest("alpha", contextLength = 100, estimateOnly = true))
                assertEquals(LoadResponse.ESTIMATED, response.status)
                assertEquals(1_200, response.estimate.totalBytes)
                assertNull(response.instance)
                assertEquals(0, engine.loads.size)
            }
        }

    @Test
    fun unloadAllAndShutdown() =
        withDaemon {
            client().use { client ->
                client.load(LoadRequest("alpha"))
                client.load(LoadRequest("alpha"))
                assertEquals(listOf(ALPHA, "$ALPHA:2"), client.unload(UnloadRequest(all = true)).unloaded)
                assertEquals(emptyList(), client.unload(UnloadRequest(all = true)).unloaded)
                client.shutdown()
                assertEquals(true, shutdownRequested())
            }
        }

    @Test
    fun withoutStreamTheAnswerIsOneJsonObject() =
        withDaemon {
            HttpClient(ClientCIO).use { http ->
                val loaded =
                    http.post(url + DaemonRoutes.LOAD) {
                        bearerAuth(TOKEN)
                        setBody("""{"model": "alpha", "some_future_field": 1}""")
                    }
                assertEquals(HttpStatusCode.OK, loaded.status)
                assertEquals(ALPHA, DaemonJson.decodeFromString(LoadResponse.serializer(), loaded.bodyAsText()).instance?.id)

                val broken =
                    http.post(url + DaemonRoutes.LOAD) {
                        bearerAuth(TOKEN)
                        setBody("{not json")
                    }
                assertEquals(HttpStatusCode.BadRequest, broken.status)
                assertEquals(ApiError.INVALID_REQUEST, DaemonJson.decodeFromString(ApiError.serializer(), broken.bodyAsText()).error.type)

                val anonymous = http.post(url + DaemonRoutes.SHUTDOWN)
                assertEquals(HttpStatusCode.Unauthorized, anonymous.status)
                assertEquals(ApiError.UNAUTHORIZED, DaemonJson.decodeFromString(ApiError.serializer(), anonymous.bodyAsText()).error.type)
            }
        }

    private companion object {
        const val TOKEN = TestDaemon.TOKEN
    }
}
