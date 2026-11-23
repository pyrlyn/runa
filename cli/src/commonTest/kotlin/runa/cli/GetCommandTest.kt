package runa.cli

import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.respondError
import io.ktor.client.request.HttpRequestData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import okio.Buffer
import okio.ByteString.Companion.toByteString
import okio.FileSystem
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class GetCommandTest {
    private val fileSystem = FileSystem.SYSTEM
    private val cli = TestCli(TestCli.temporaryHome())
    private val directory = cli.home / "models" / "acme" / "tiny-GGUF"
    private val requests = mutableListOf<HttpRequestData>()

    // Padded to different sizes, so "largest" and "smallest" mean something.
    private fun model(
        fileType: Long,
        padding: Int,
    ): ByteArray =
        Buffer()
            .also { tinyLlama(fileType).put("runa.test.padding", "x".repeat(padding)).writeTo(it) }
            .readByteArray()

    private val files =
        mapOf(
            "README.md" to "# tiny".encodeToByteArray(),
            "tiny-Q4_K_M.gguf" to model(fileType = 15, padding = 100),
            "tiny-Q8_0.gguf" to model(fileType = 7, padding = 3_000),
            "tiny-f16.gguf" to model(fileType = 1, padding = 9_000),
            "mmproj-tiny-f16.gguf" to model(fileType = 1, padding = 10),
        )

    @AfterTest
    fun cleanUp() = fileSystem.deleteRecursively(cli.home)

    /** A Hub with one repository, `acme/tiny-GGUF`, holding [files]. */
    private fun hub(files: Map<String, ByteArray> = this.files) =
        MockEngine { request ->
            requests += request
            val path = request.url.encodedPath
            when {
                path == "/api/models/acme/tiny-GGUF/tree/main" -> {
                    respond(
                        files.entries.joinToString(",", "[", "]") { (name, bytes) ->
                            val sha256 = bytes.toByteString().sha256().hex()
                            """{"type":"file","path":"$name","size":134,"lfs":{"oid":"$sha256","size":${bytes.size}}}"""
                        },
                    )
                }

                path.startsWith("/acme/tiny-GGUF/resolve/main/") -> {
                    respond(files.getValue(path.substringAfterLast('/')))
                }

                else -> {
                    respondError(HttpStatusCode.Unauthorized)
                }
            }
        }

    @Test
    fun downloadsTheLargestQuantizedVariantByDefault() {
        val run = cli.run("get", "acme/tiny-GGUF", http = hub())
        assertEquals(
            "acme/tiny-GGUF has 3 variants; taking Q8_0 (--select lists them)\n" +
                "Downloading acme/tiny-GGUF@Q8_0 (3.27 KB) to $directory\n" +
                "Downloaded acme/tiny-GGUF@q8_0\n",
            run.stderr,
        )
        assertEquals("${directory / "tiny-Q8_0.gguf"}\n", run.stdout)
        assertEquals(0, run.exitCode)
        assertContentEquals(files.getValue("tiny-Q8_0.gguf"), fileSystem.read(directory / "tiny-Q8_0.gguf") { readByteArray() })
        assertEquals(listOf("tiny-Q8_0.gguf"), fileSystem.list(directory).map { it.name })
        assertContains(cli.run("ls", "--variants").stdout, "acme/tiny-GGUF@q8_0")
    }

    @Test
    fun takesTheSmallestVariantWhenNothingFitsTheMemory() {
        val run = cli.run("get", "acme/tiny-GGUF", http = hub(), memory = 100)
        assertEquals(0, run.exitCode)
        assertContains(run.stderr, "taking Q4_K_M")
        assertContains(run.stderr, "Warning: Q4_K_M is ")
        assertContains(run.stderr, "this machine has 100 B of memory, so it may not load")
        assertTrue(fileSystem.exists(directory / "tiny-Q4_K_M.gguf"))
    }

    @Test
    fun theQuantizationAfterTheAtSignIsTaken() {
        val run = cli.run("get", "acme/tiny-GGUF@q4_k_m", http = hub())
        assertEquals(0, run.exitCode)
        assertEquals("${directory / "tiny-Q4_K_M.gguf"}\n", run.stdout)

        val missing = cli.run("get", "acme/tiny-GGUF@q5_k_m", http = hub())
        assertEquals(2, missing.exitCode)
        assertEquals("Error: acme/tiny-GGUF has no Q5_K_M variant; it has: Q4_K_M, Q8_0, F16\n", missing.stderr)
    }

    @Test
    fun aUrlOfOneFileDownloadsThatFile() {
        val run = cli.run("get", "https://huggingface.co/acme/tiny-GGUF/blob/main/tiny-f16.gguf", http = hub())
        assertEquals(0, run.exitCode)
        assertEquals("${directory / "tiny-f16.gguf"}\n", run.stdout)

        val notAModel = cli.run("get", "https://huggingface.co/acme/tiny-GGUF/blob/main/mmproj-tiny-f16.gguf", http = hub())
        assertEquals(2, notAModel.exitCode)
        assertEquals("Error: acme/tiny-GGUF has no GGUF model file mmproj-tiny-f16.gguf\n", notAModel.stderr)
    }

    @Test
    fun selectAsksWhichVariantWhenSomeoneCanAnswer() {
        val asked = cli.run("get", "acme/tiny-GGUF", "--select", http = hub(), input = listOf("9", "1"))
        assertEquals(0, asked.exitCode)
        assertEquals(
            """
            |#  VARIANT  SIZE              
            |1  Q4_K_M   374 B             
            |2  Q8_0     3.27 KB  default  
            |3  F16      9.27 KB           
            |Variant (2): Invalid value, choose from [1, 2, 3]
            |Variant (2): ${directory / "tiny-Q4_K_M.gguf"}
            |
            """.trimMargin(),
            asked.stdout,
        )
        assertTrue(fileSystem.exists(directory / "tiny-Q4_K_M.gguf"))

        val yes = cli.run("-y", "get", "acme/tiny-GGUF", "--select", http = hub(), input = listOf("3"))
        assertEquals("${directory / "tiny-Q8_0.gguf"}\n", yes.stdout)
        assertFalse(fileSystem.exists(directory / "tiny-f16.gguf"))
    }

    @Test
    fun printsJson() {
        val run = cli.run("get", "acme/tiny-GGUF@Q4_K_M", "--json", http = hub())
        val size = files.getValue("tiny-Q4_K_M.gguf").size
        assertEquals(
            "{\"key\":\"acme/tiny-GGUF@q4_k_m\",\"repo\":\"acme/tiny-GGUF\",\"revision\":\"main\",\"quantization\":\"Q4_K_M\"," +
                "\"size_bytes\":$size,\"downloaded\":true,\"files\":[\"${directory / "tiny-Q4_K_M.gguf"}\"]}\n",
            run.stdout,
        )
        assertEquals(0, run.exitCode)
    }

    @Test
    fun aModelThatIsAlreadyThereIsNotDownloadedAgain() {
        assertEquals(0, cli.run("get", "acme/tiny-GGUF@Q4_K_M", http = hub()).exitCode)
        requests.clear()
        val again = cli.run("get", "acme/tiny-GGUF@Q4_K_M", "--json", http = hub())
        assertEquals(0, again.exitCode)
        assertContains(again.stderr, "acme/tiny-GGUF@q4_k_m is already in the model store\n")
        assertContains(again.stdout, "\"downloaded\":false")
        assertEquals(listOf("/api/models/acme/tiny-GGUF/tree/main"), requests.map { it.url.encodedPath })
    }

    @Test
    fun theTokenComesFromTheEnvironment() {
        cli.run("get", "acme/tiny-GGUF@Q4_K_M", http = hub(), env = mapOf("HF_TOKEN" to "secret"))
        assertEquals(listOf<String?>("Bearer secret", "Bearer secret"), requests.map { it.headers[HttpHeaders.Authorization] })
    }

    @Test
    fun aGatedModelWithoutTokenSaysWhatToDo() {
        val gated =
            MockEngine { request ->
                if (request.url.encodedPath.startsWith("/api/")) {
                    respond("""[{"type":"file","path":"tiny-Q4_K_M.gguf","size":100}]""")
                } else {
                    respondError(HttpStatusCode.Unauthorized, headers = headersOf("X-Error-Code", "GatedRepo"))
                }
            }
        val run = cli.run("get", "acme/tiny-GGUF@Q4_K_M", http = gated)
        assertEquals(2, run.exitCode)
        assertContains(
            run.stderr,
            "Error: acme/tiny-GGUF is gated: accept its terms at https://huggingface.co/acme/tiny-GGUF, " +
                "then set HF_TOKEN to an access token of that account\n",
        )
        assertEquals("", run.stdout)
    }

    @Test
    fun mistakesAndFailuresHaveTheirOwnExitCodes() {
        val notAName = cli.run("get", "tiny", http = hub())
        assertEquals(2, notAName.exitCode)
        assertContains(notAName.stderr, "'tiny' is not a Hugging Face model")

        val unknown = cli.run("get", "acme/other", http = hub())
        assertEquals(2, unknown.exitCode)
        assertEquals("Error: acme/other was not found on huggingface.co; if it is private, set HF_TOKEN\n", unknown.stderr)

        val noModels = cli.run("get", "acme/tiny-GGUF", http = hub(files.filterKeys { !it.startsWith("tiny") }))
        assertEquals(2, noModels.exitCode)
        assertContains(noModels.stderr, "acme/tiny-GGUF has no GGUF model files")

        val down = cli.run("get", "acme/tiny-GGUF", http = MockEngine { respondError(HttpStatusCode.ServiceUnavailable) })
        assertEquals(3, down.exitCode)
        assertContains(down.stderr, "huggingface.co answered 503")

        val plainText = cli.run("get", "acme/tiny-GGUF", http = hub(), env = mapOf("HF_ENDPOINT" to "http://mirror.example.com"))
        assertEquals(2, plainText.exitCode)
        assertContains(plainText.stderr, "HF_ENDPOINT must be an https:// URL")
        assertFalse(fileSystem.exists(cli.home / "models"))
    }

    @Test
    fun aMirrorIsUsedWhenTheEndpointIsSet() {
        val run = cli.run("get", "acme/tiny-GGUF@Q4_K_M", http = hub(), env = mapOf("HF_ENDPOINT" to "https://mirror.example.com"))
        assertEquals(0, run.exitCode)
        assertEquals(listOf("mirror.example.com", "mirror.example.com"), requests.map { it.url.host })
    }

    @Test
    fun aDownloadThatIsNotAModelIsReported() {
        val run = cli.run("get", "acme/tiny-GGUF", http = hub(mapOf("tiny-Q4_K_M.gguf" to "<html>".encodeToByteArray())))
        assertEquals(0, run.exitCode)
        assertContains(run.stderr, "Warning: tiny-Q4_K_M.gguf was downloaded, but the model store cannot use it: Not a GGUF file")
        assertContains(
            cli.run("get", "acme/tiny-GGUF", "--json", http = hub(mapOf("tiny-Q4_K_M.gguf" to "<html>".encodeToByteArray()))).stdout,
            "\"key\":null",
        )
    }
}
