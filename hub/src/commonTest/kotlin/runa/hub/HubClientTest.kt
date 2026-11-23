package runa.hub

import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.respondError
import io.ktor.client.request.HttpRequestData
import io.ktor.client.request.HttpResponseData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.runBlocking
import okio.ByteString.Companion.toByteString
import okio.FileSystem
import okio.IOException
import okio.Path
import runa.core.BackendError
import runa.core.UserInputError
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HubClientTest {
    private val fileSystem = FileSystem.SYSTEM
    private val root: Path = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-hub-test-${Random.nextLong().toULong()}"
    private val repo = HubRepo("acme", "tiny-GGUF")
    private val content = Random(7).nextBytes(200_000)
    private val file = HubFile("tiny-Q4_K_M.gguf", content.size.toLong(), content.toByteString().sha256().hex())
    private val destination = root / "acme" / "tiny-GGUF" / file.name
    private val partial = root / "acme" / "tiny-GGUF" / "${file.name}.part"
    private val requests = mutableListOf<HttpRequestData>()

    @AfterTest
    fun cleanUp() = fileSystem.deleteRecursively(root)

    private fun client(
        token: String? = null,
        handler: suspend MockRequestHandleScope.(HttpRequestData) -> HttpResponseData,
    ) = HubClient(
        MockEngine { request ->
            requests += request
            handler(request)
        },
        token = token,
    )

    private fun <T> HubClient.run(block: suspend HubClient.() -> T): T = use { runBlocking { it.block() } }

    /** Serves [content], honouring `Range` unless [ignoreRange]; sends only [limit] bytes of the answer when set. */
    private fun MockRequestHandleScope.serve(
        request: HttpRequestData,
        ignoreRange: Boolean = false,
        limit: Int? = null,
    ): HttpResponseData {
        val from =
            request.headers[HttpHeaders.Range]
                ?.takeUnless { ignoreRange }
                ?.removePrefix("bytes=")
                ?.removeSuffix("-")
                ?.toInt()
        val body = content.copyOfRange(from ?: 0, content.size)
        return respond(
            body.copyOf(minOf(limit ?: body.size, body.size)),
            if (from == null) HttpStatusCode.OK else HttpStatusCode.PartialContent,
        )
    }

    private val tree =
        """
        [
          {"type":"directory","oid":"1","size":0,"path":"sub"},
          {"type":"file","oid":"2","size":1519,"path":"README.md"},
          {"type":"file","oid":"3","size":134,"path":"tiny-Q4_K_M.gguf",
           "lfs":{"oid":"${"AB".repeat(32)}","size":200000,"pointerSize":134},"xetHash":"x"},
          {"type":"file","oid":"4","size":10,"path":"sub/..\\evil.gguf"},
          {"type":"file","oid":"5","size":10,"path":".hidden.gguf"},
          {"type":"file","oid":"6","size":10,"path":"bad-hash.gguf","lfs":{"oid":"nope","size":77}}
        ]
        """.trimIndent()

    @Test
    fun listsTheFilesOfARepository() {
        val files = client(token = "secret") { respond(tree) }.run { listFiles(repo) }
        assertEquals(
            listOf(
                HubFile("README.md", 1519, null),
                HubFile("tiny-Q4_K_M.gguf", 200_000, "ab".repeat(32)),
                HubFile("bad-hash.gguf", 77, null),
            ),
            files,
        )
        val request = requests.single()
        assertEquals("https://huggingface.co/api/models/acme/tiny-GGUF/tree/main?recursive=true", request.url.toString())
        assertEquals("Bearer secret", request.headers[HttpHeaders.Authorization])
    }

    @Test
    fun followsThePagesOfALongList() {
        val files =
            client { request ->
                if (request.url.parameters["cursor"] == null) {
                    respond(
                        """[{"type":"file","size":1,"path":"a.gguf"}]""",
                        headers =
                            headersOf(
                                HttpHeaders.Link,
                                "<https://huggingface.co/api/models/acme/tiny-GGUF/tree/main?recursive=true&cursor=abc>; rel=\"next\"",
                            ),
                    )
                } else {
                    respond("""[{"type":"file","size":2,"path":"b.gguf"}]""")
                }
            }.run { listFiles(repo) }
        assertEquals(listOf("a.gguf", "b.gguf"), files.map { it.path })
        assertEquals(listOf(null, "abc"), requests.map { it.url.parameters["cursor"] })
    }

    @Test
    fun aGatedRepositoryWithoutTokenAsksForOne() {
        val error =
            assertFailsWith<UserInputError> {
                client { respondError(HttpStatusCode.Unauthorized, headers = headersOf("X-Error-Code", "GatedRepo")) }
                    .run { download(repo, file, destination, fileSystem) }
            }
        assertContains(error.message.orEmpty(), "acme/tiny-GGUF is gated")
        assertContains(error.message.orEmpty(), "https://huggingface.co/acme/tiny-GGUF")
        assertContains(error.message.orEmpty(), "HF_TOKEN")
        assertFalse(fileSystem.exists(partial))
    }

    @Test
    fun aMissingRepositoryIsTheUsersMistake() {
        val error = assertFailsWith<UserInputError> { client { respondError(HttpStatusCode.Unauthorized) }.run { listFiles(repo) } }
        assertEquals("acme/tiny-GGUF was not found on huggingface.co; if it is private, set HF_TOKEN", error.message)
    }

    @Test
    fun aMissingRevisionIsNamed() {
        val error =
            assertFailsWith<UserInputError> {
                client { respondError(HttpStatusCode.NotFound, headers = headersOf("X-Error-Code", "RevisionNotFound")) }
                    .run { listFiles(repo.copy(revision = "v9")) }
            }
        assertEquals("acme/tiny-GGUF has no revision 'v9'", error.message)
    }

    @Test
    fun hubFailuresAreBackendErrors() {
        assertFailsWith<BackendError> { client { respondError(HttpStatusCode.BadGateway) }.run { listFiles(repo) } }
        assertFailsWith<BackendError> { client { respondError(HttpStatusCode.TooManyRequests) }.run { listFiles(repo) } }
        assertFailsWith<BackendError> { client { respond("<html>") }.run { listFiles(repo) } }
        val error = assertFailsWith<BackendError> { client { throw IOException("no route") }.run { listFiles(repo) } }
        assertContains(error.message.orEmpty(), "Cannot reach huggingface.co")
    }

    @Test
    fun downloadsAndVerifiesAFile() {
        var progress = 0L
        val fetched = client { serve(it) }.run { download(repo, file, destination, fileSystem) { progress += it } }
        assertTrue(fetched)
        assertContentEquals(content, fileSystem.read(destination) { readByteArray() })
        assertFalse(fileSystem.exists(partial))
        assertEquals(content.size.toLong(), progress)
        assertEquals("https://huggingface.co/acme/tiny-GGUF/resolve/main/tiny-Q4_K_M.gguf", requests.single().url.toString())
        assertNull(requests.single().headers[HttpHeaders.Range])

        requests.clear()
        assertFalse(client { serve(it) }.run { download(repo, file, destination, fileSystem) })
        assertEquals(emptyList(), requests)
    }

    @Test
    fun resumesAfterACutConnection() {
        var progress = 0L
        val error =
            assertFailsWith<BackendError> {
                client { serve(it, limit = 70_000) }.run { download(repo, file, destination, fileSystem) { progress += it } }
            }
        assertContains(error.message.orEmpty(), "stopped at 70000 of 200000 bytes")
        assertEquals(70_000L, fileSystem.metadata(partial).size)
        assertFalse(fileSystem.exists(destination))

        assertTrue(client { serve(it) }.run { download(repo, file, destination, fileSystem) { progress += it } })
        assertEquals("bytes=70000-", requests.last().headers[HttpHeaders.Range])
        assertContentEquals(content, fileSystem.read(destination) { readByteArray() })
        assertFalse(fileSystem.exists(partial))
        assertEquals(70_000L + 200_000L, progress)
    }

    @Test
    fun startsOverWhenTheServerIgnoresTheRange() {
        fileSystem.createDirectories(checkNotNull(partial.parent))
        fileSystem.write(partial) { write(content, 0, 50_000) }
        var progress = 0L
        assertTrue(
            client { serve(it, ignoreRange = true) }.run { download(repo, file, destination, fileSystem) { progress += it } },
        )
        assertEquals("bytes=50000-", requests.single().headers[HttpHeaders.Range])
        assertContentEquals(content, fileSystem.read(destination) { readByteArray() })
        assertEquals(content.size.toLong(), progress)
    }

    @Test
    fun aPartialFileThatIsAlreadyCompleteIsOnlyVerified() {
        fileSystem.createDirectories(checkNotNull(partial.parent))
        fileSystem.write(partial) { write(content) }
        assertTrue(client { serve(it) }.run { download(repo, file, destination, fileSystem) })
        assertEquals(emptyList(), requests)
        assertContentEquals(content, fileSystem.read(destination) { readByteArray() })
    }

    @Test
    fun aChecksumMismatchRemovesTheDownload() {
        val wrong = file.copy(sha256 = "0".repeat(64))
        val error = assertFailsWith<BackendError> { client { serve(it) }.run { download(repo, wrong, destination, fileSystem) } }
        assertContains(error.message.orEmpty(), "does not match its checksum")
        assertFalse(fileSystem.exists(partial))
        assertFalse(fileSystem.exists(destination))
    }

    @Test
    fun aCorruptPartialFileIsCaughtByTheChecksum() {
        fileSystem.createDirectories(checkNotNull(partial.parent))
        fileSystem.write(partial) { write(ByteArray(50_000)) }
        assertFailsWith<BackendError> { client { serve(it) }.run { download(repo, file, destination, fileSystem) } }
        assertFalse(fileSystem.exists(partial))
        assertFalse(fileSystem.exists(destination))
    }

    @Test
    fun aBodyLongerThanListedIsRefused() {
        val shorter = file.copy(size = 1_000, sha256 = null)
        val error = assertFailsWith<BackendError> { client { serve(it) }.run { download(repo, shorter, destination, fileSystem) } }
        assertContains(error.message.orEmpty(), "larger on the server")
        assertFalse(fileSystem.exists(destination))
    }

    @Test
    fun anExistingFileOfAnotherSizeIsNotOverwritten() {
        fileSystem.createDirectories(checkNotNull(destination.parent))
        fileSystem.write(destination) { writeUtf8("mine") }
        assertFailsWith<UserInputError> { client { serve(it) }.run { download(repo, file, destination, fileSystem) } }
        assertEquals("mine", fileSystem.read(destination) { readUtf8() })
        assertEquals(emptyList(), requests)
    }

    @Test
    fun theTokenIsNotSentToTheHostARedirectLeadsTo() {
        val fetched =
            client(token = "secret") { request ->
                when {
                    request.url.host == "cdn.example.com" -> {
                        serve(request)
                    }

                    request.url.encodedPath.startsWith("/api/resolve-cache/") -> {
                        respond("", HttpStatusCode.Found, headersOf(HttpHeaders.Location, "https://cdn.example.com/blob?sig=1"))
                    }

                    else -> {
                        respond(
                            "",
                            HttpStatusCode.TemporaryRedirect,
                            headersOf(HttpHeaders.Location, "/api/resolve-cache/models/acme/tiny-GGUF/abc/tiny-Q4_K_M.gguf"),
                        )
                    }
                }
            }.run { download(repo, file, destination, fileSystem) }
        assertTrue(fetched)
        assertEquals(
            listOf(
                "https://huggingface.co/acme/tiny-GGUF/resolve/main/tiny-Q4_K_M.gguf",
                "https://huggingface.co/api/resolve-cache/models/acme/tiny-GGUF/abc/tiny-Q4_K_M.gguf",
                "https://cdn.example.com/blob?sig=1",
            ),
            requests.map { it.url.toString() },
        )
        assertEquals(listOf("Bearer secret", "Bearer secret", null), requests.map { it.headers[HttpHeaders.Authorization] })
        assertContentEquals(content, fileSystem.read(destination) { readByteArray() })
    }

    @Test
    fun aRedirectLoopEnds() {
        val error =
            assertFailsWith<BackendError> {
                client { respond("", HttpStatusCode.Found, headersOf(HttpHeaders.Location, "/again")) }.run { listFiles(repo) }
            }
        assertContains(error.message.orEmpty(), "redirects in a loop")
        assertEquals(5, requests.size)
    }

    @Test
    fun anEndpointMustNotCarryTheTokenInClearText() {
        val engine =
            MockEngine {
                requests += it
                respond("[]")
            }
        listOf("http://mirror.example.com", "mirror.example.com", "ftp://mirror.example.com", "").forEach {
            assertFailsWith<UserInputError>(it) { HubClient(engine, endpoint = it) }
        }
        HubClient(engine, endpoint = "http://127.0.0.1:8080").run { listFiles(repo) }
        HubClient(engine, endpoint = "https://mirror.example.com/").run { listFiles(repo) }
        assertEquals(
            listOf(
                "http://127.0.0.1:8080/api/models/acme/tiny-GGUF/tree/main?recursive=true",
                "https://mirror.example.com/api/models/acme/tiny-GGUF/tree/main?recursive=true",
            ),
            requests.map { it.url.toString() },
        )
    }

    @Test
    fun downloadsEveryPartOfASplitModel() {
        val parts =
            listOf(0 to 120_000, 120_000 to 200_000).mapIndexed { index, (from, to) ->
                val bytes = content.copyOfRange(from, to)
                HubFile("big-Q8_0-0000${index + 1}-of-00002.gguf", bytes.size.toLong(), bytes.toByteString().sha256().hex()) to bytes
            }
        val variant = variantsOf(parts.map { it.first }).single()
        val directory = root / "acme" / "tiny-GGUF"
        val (paths, fetched) =
            client { request -> respond(parts.first { request.url.encodedPath.endsWith(it.first.name) }.second) }
                .run { download(repo, variant, directory, fileSystem) }
        assertTrue(fetched)
        assertEquals(parts.map { directory / it.first.name }, paths)
        parts.forEach { (part, bytes) -> assertContentEquals(bytes, fileSystem.read(directory / part.name) { readByteArray() }) }
    }
}
