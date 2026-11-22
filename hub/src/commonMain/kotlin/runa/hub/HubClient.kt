package runa.hub

import io.ktor.client.HttpClient
import io.ktor.client.HttpClientConfig
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.header
import io.ktor.client.request.prepareGet
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsChannel
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.URLBuilder
import io.ktor.http.Url
import io.ktor.http.appendPathSegments
import io.ktor.http.takeFrom
import io.ktor.utils.io.readAvailable
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import okio.FileSystem
import okio.HashingSink
import okio.IOException
import okio.Path
import okio.blackholeSink
import okio.buffer
import okio.use
import runa.core.BackendError
import runa.core.UserInputError

/**
 * Lists and downloads model files from the Hugging Face Hub (or a mirror at [endpoint]).
 *
 * @param engine null for the platform's own HTTP engine; tests pass a mock.
 * @param token sent to [endpoint] only, never to the hosts it redirects downloads to.
 */
public class HubClient(
    engine: HttpClientEngine? = null,
    endpoint: String = DEFAULT_ENDPOINT,
    private val token: String? = null,
) : AutoCloseable {
    private val endpoint = Url(endpoint)
    private val http = if (engine != null) HttpClient(engine) { configure() } else HttpClient { configure() }

    private fun HttpClientConfig<*>.configure() {
        // Redirects are followed by hand, so the token can be dropped when the host changes.
        followRedirects = false
        install(HttpTimeout) {
            connectTimeoutMillis = CONNECT_TIMEOUT_MILLIS
            socketTimeoutMillis = SOCKET_TIMEOUT_MILLIS
        }
    }

    /**
     * Every file of [repo] at its revision.
     *
     * @throws UserInputError when the repository or revision does not exist or cannot be seen.
     * @throws BackendError when the Hub cannot be reached or fails.
     */
    public suspend fun listFiles(repo: HubRepo): List<HubFile> {
        val files = mutableListOf<HubFile>()
        var page: Url? =
            URLBuilder(endpoint)
                .apply {
                    appendPathSegments("api", "models", repo.owner, repo.name, "tree", repo.revision)
                    parameters.append("recursive", "true")
                }.build()
        var pages = 0
        while (page != null) {
            if (++pages > MAX_PAGES) throw BackendError("${repo.id} lists more files than runa reads")
            page =
                request(page, resumeFrom = null) { response ->
                    if (response.status != HttpStatusCode.OK) throw failure(response, repo, "the file list")
                    files += parseTree(response.bodyAsText(), repo)
                    nextPage(response.headers[HttpHeaders.Link])
                }
        }
        return files
    }

    /**
     * Downloads [file] to [destination] through `<destination>.part`, which is kept when the
     * download stops early and continued on the next call. The file appears under its real name
     * only after its size and SHA-256 are checked.
     *
     * [onProgress] gets the bytes gained since its last call, on the caller's coroutine; the bytes
     * of an earlier partial download come first, and a negative number means a restart.
     *
     * @return false when [destination] was already complete.
     */
    public suspend fun download(
        repo: HubRepo,
        file: HubFile,
        destination: Path,
        fileSystem: FileSystem,
        onProgress: (Long) -> Unit = {},
    ): Boolean {
        val partial = checkNotNull(destination.parent) / "${destination.name}$PARTIAL_SUFFIX"
        val transfer = Transfer(file, partial, fileSystem, onProgress)
        try {
            val present = fileSystem.metadataOrNull(destination)
            if (present != null) {
                if (present.size == file.size) return false
                throw UserInputError("$destination exists with a different size than ${file.name} on the Hub; remove it to download again")
            }
            fileSystem.createDirectories(checkNotNull(destination.parent))
            transfer.resume()
        } catch (e: IOException) {
            throw UserInputError("Cannot write $partial: ${e.message}")
        }
        if (transfer.received < file.size) {
            val url =
                URLBuilder(endpoint)
                    .apply {
                        appendPathSegments(listOf(repo.owner, repo.name, "resolve", repo.revision) + file.path.split('/'))
                    }.build()
            request(url, resumeFrom = transfer.received.takeIf { it > 0 }) { response ->
                when (response.status) {
                    HttpStatusCode.PartialContent -> transfer.receive(response, append = true)

                    // A server that ignores Range sends the whole file again.
                    HttpStatusCode.OK -> transfer.receive(response, append = false)

                    else -> throw failure(response, repo, file.name)
                }
            }
        }
        try {
            transfer.finish(destination)
        } catch (e: IOException) {
            throw UserInputError("Cannot write $destination: ${e.message}")
        }
        return true
    }

    /**
     * Downloads every file of [variant] into [directory], a few at a time.
     *
     * @return the downloaded files, and whether anything had to be fetched.
     */
    public suspend fun download(
        repo: HubRepo,
        variant: HubVariant,
        directory: Path,
        fileSystem: FileSystem,
        onProgress: (Long) -> Unit = {},
    ): Pair<List<Path>, Boolean> =
        coroutineScope {
            val slots = Semaphore(PARALLEL_DOWNLOADS)
            val fetched =
                variant.files
                    .map { file -> async { slots.withPermit { download(repo, file, directory / file.name, fileSystem, onProgress) } } }
                    .awaitAll()
            variant.files.map { directory / it.name } to fetched.any { it }
        }

    override fun close(): Unit = http.close()

    /** One GET, following redirects; [handle] sees the first response that is not a redirect. */
    private suspend fun <T> request(
        url: Url,
        resumeFrom: Long?,
        handle: suspend (HttpResponse) -> T,
    ): T {
        var current = url
        repeat(MAX_REDIRECTS) {
            var result: Result<T>? = null
            val next =
                try {
                    http
                        .prepareGet(current) {
                            if (token != null && current.sameOriginAs(endpoint)) header(HttpHeaders.Authorization, "Bearer $token")
                            if (resumeFrom != null) header(HttpHeaders.Range, "bytes=$resumeFrom-")
                        }.execute { response ->
                            val location = response.headers[HttpHeaders.Location]
                            if (response.status.value in REDIRECT_STATUSES && location != null) {
                                URLBuilder(current).apply { parameters.clear() }.takeFrom(location).build()
                            } else {
                                result = Result.success(handle(response))
                                null
                            }
                        }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: UserInputError) {
                    throw e
                } catch (e: BackendError) {
                    throw e
                } catch (e: Exception) {
                    throw BackendError("Cannot reach ${current.host}: ${e.message ?: e::class.simpleName}", e)
                }
            result?.let { return it.getOrThrow() }
            current = checkNotNull(next)
        }
        throw BackendError("${url.host} redirects in a loop")
    }

    private fun failure(
        response: HttpResponse,
        repo: HubRepo,
        what: String,
    ): RuntimeException {
        val page = "${endpoint.toString().trimEnd('/')}/${repo.id}"
        return when {
            response.headers[ERROR_CODE] == "GatedRepo" -> {
                UserInputError(
                    if (token == null) {
                        "${repo.id} is gated: accept its terms at $page, then set HF_TOKEN to an access token of that account"
                    } else {
                        "${repo.id} is gated and HF_TOKEN has no access to it: accept its terms at $page"
                    },
                )
            }

            response.headers[ERROR_CODE] == "RevisionNotFound" -> {
                UserInputError("${repo.id} has no revision '${repo.revision}'")
            }

            // The Hub answers 401 for a repository that does not exist, so that private ones stay hidden.
            response.status.value in NOT_VISIBLE_STATUSES -> {
                UserInputError(
                    if (token == null) {
                        "${repo.id} was not found on ${endpoint.host}; if it is private, set HF_TOKEN"
                    } else {
                        "${repo.id} was not found on ${endpoint.host}, or HF_TOKEN has no access to it"
                    },
                )
            }

            response.status == HttpStatusCode.TooManyRequests -> {
                BackendError("${endpoint.host} is limiting requests; try again later")
            }

            else -> {
                BackendError("${endpoint.host} answered ${response.status.value} for $what of ${repo.id}")
            }
        }
    }

    private fun parseTree(
        body: String,
        repo: HubRepo,
    ): List<HubFile> {
        val entries =
            try {
                json.decodeFromString(ListSerializer(TreeEntry.serializer()), body)
            } catch (e: SerializationException) {
                throw BackendError("${endpoint.host} sent a file list for ${repo.id} that runa cannot read")
            } catch (e: IllegalArgumentException) {
                throw BackendError("${endpoint.host} sent a file list for ${repo.id} that runa cannot read")
            }
        return entries.mapNotNull { entry ->
            val size = entry.lfs?.size ?: entry.size
            // The list is remote input and the names become local files: anything odd is left out.
            if (entry.type != "file" || size < 0 || !isSafeFileName(entry.path.substringAfterLast('/'))) return@mapNotNull null
            HubFile(
                entry.path,
                size,
                entry.lfs
                    ?.oid
                    ?.lowercase()
                    ?.takeIf(SHA256::matches),
            )
        }
    }

    private fun nextPage(link: String?): Url? =
        link?.let(NEXT_LINK::find)?.groupValues?.get(1)?.let {
            URLBuilder(endpoint).apply { parameters.clear() }.takeFrom(it).build()
        }

    /** The state of one file's download: what is on disk so far and its running hash. */
    private class Transfer(
        private val file: HubFile,
        private val partial: Path,
        private val fileSystem: FileSystem,
        private val onProgress: (Long) -> Unit,
    ) {
        var received = 0L
            private set
        private var hash = HashingSink.sha256(blackholeSink())

        /** Picks up an earlier partial download, hashing what is already there. */
        fun resume() {
            val size = fileSystem.metadataOrNull(partial)?.size ?: return
            if (size > file.size) {
                fileSystem.delete(partial)
                return
            }
            fileSystem.source(partial).use { hash.buffer().apply { writeAll(it) }.flush() }
            received = size
            onProgress(size)
        }

        suspend fun receive(
            response: HttpResponse,
            append: Boolean,
        ) {
            if (!append && received > 0) {
                onProgress(-received)
                received = 0
                hash = HashingSink.sha256(blackholeSink())
            }
            val channel = response.bodyAsChannel()
            val chunk = ByteArray(CHUNK_BYTES)
            val hashed = hash.buffer()
            try {
                (if (append) fileSystem.appendingSink(partial) else fileSystem.sink(partial)).buffer().use { out ->
                    while (true) {
                        val count =
                            try {
                                channel.readAvailable(chunk, 0, chunk.size)
                            } catch (e: CancellationException) {
                                throw e
                            } catch (e: Exception) {
                                throw interrupted(e.message)
                            }
                        if (count < 0) break
                        if (received + count > file.size) throw BackendError("${file.name} is larger on the server than the Hub listed")
                        out.write(chunk, 0, count)
                        hashed.write(chunk, 0, count)
                        received += count
                        onProgress(count.toLong())
                    }
                }
            } catch (e: IOException) {
                throw UserInputError("Cannot write $partial: ${e.message}")
            } finally {
                hashed.flush()
            }
            if (received < file.size) throw interrupted("the connection closed")
        }

        fun finish(destination: Path) {
            if (received != file.size) throw interrupted("the connection closed")
            if (file.sha256 != null && hash.hash.hex() != file.sha256) {
                // Resuming would only repeat the mismatch, so the next attempt has to start clean.
                fileSystem.delete(partial)
                throw BackendError("${file.name} does not match its checksum on the Hub; the download was removed, run the command again")
            }
            fileSystem.atomicMove(partial, destination)
        }

        private fun interrupted(reason: String?) =
            BackendError(
                "The download of ${file.name} stopped at $received of ${file.size} bytes (${reason ?: "connection lost"}); run the command again to continue",
            )
    }

    @Serializable
    private data class TreeEntry(
        val type: String = "",
        val path: String = "",
        val size: Long = -1,
        val lfs: Lfs? = null,
    )

    @Serializable
    private data class Lfs(
        val oid: String? = null,
        val size: Long? = null,
    )

    public companion object {
        public const val DEFAULT_ENDPOINT: String = "https://huggingface.co"

        /** Appended to a file name while it is being downloaded. */
        public const val PARTIAL_SUFFIX: String = ".part"

        private const val PARALLEL_DOWNLOADS = 4
        private const val MAX_REDIRECTS = 5
        private const val MAX_PAGES = 50
        private const val CHUNK_BYTES = 64 * 1024
        private const val CONNECT_TIMEOUT_MILLIS = 30_000L
        private const val SOCKET_TIMEOUT_MILLIS = 60_000L
        private const val ERROR_CODE = "X-Error-Code"
        private val REDIRECT_STATUSES = setOf(301, 302, 303, 307, 308)
        private val NOT_VISIBLE_STATUSES = setOf(401, 403, 404)
        private val SHA256 = Regex("[0-9a-f]{64}")
        private val NEXT_LINK = Regex("<([^>]+)>\\s*;\\s*rel=\"next\"")
        private val json = Json { ignoreUnknownKeys = true }

        private fun isSafeFileName(name: String): Boolean =
            name.isNotEmpty() && !name.startsWith(".") && name.none { it == '\\' || it == ':' || it.isISOControl() }
    }
}

private fun Url.sameOriginAs(other: Url): Boolean = protocol == other.protocol && host == other.host && port == other.port
