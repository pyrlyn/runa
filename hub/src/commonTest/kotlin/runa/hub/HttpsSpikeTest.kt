package runa.hub

import io.ktor.client.HttpClient
import io.ktor.client.request.get
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpStatusCode
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Phase 0 spike (PLAN.md): the platform's default Ktor engine reaches Hugging Face over HTTPS.
class HttpsSpikeTest {
    @Test
    fun getsOverHttps() =
        runBlocking {
            HttpClient().use { client ->
                val response = client.get("https://huggingface.co/api/models?limit=1")
                assertEquals(HttpStatusCode.OK, response.status)
                assertTrue(response.bodyAsText().startsWith("["))
            }
        }
}
