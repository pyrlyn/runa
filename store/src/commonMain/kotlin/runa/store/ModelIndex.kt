package runa.store

import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import okio.FileSystem
import okio.IOException
import okio.Path
import runa.gguf.GgufHeader
import runa.platform.writeAtomically

/** What the store remembers about one `.gguf` file, valid while its size and mtime are unchanged. */
@Serializable
internal data class IndexEntry(
    val size: Long,
    val mtime: Long,
    /** `general.type`; anything but null or `model` is not listed (projectors, adapters). */
    val type: String? = null,
    val architecture: String? = null,
    val name: String? = null,
    val parameters: Long = 0,
    val quantization: String? = null,
    val contextLength: Long? = null,
    val embedding: Boolean = false,
) {
    constructor(size: Long, mtime: Long, header: GgufHeader) : this(
        size = size,
        mtime = mtime,
        type = header.type,
        architecture = header.architecture,
        name = header.name,
        parameters = header.parameterCount,
        quantization = header.quantization,
        contextLength = header.contextLength,
        embedding = header.isEmbeddingModel,
    )
}

/**
 * The metadata cache at `<models dir>/.index.json`, keyed by path relative to the models dir. It
 * is only a cache: a missing, unreadable or outdated file means every header is read again.
 */
@Serializable
internal data class ModelIndex(
    val version: Int = VERSION,
    val files: Map<String, IndexEntry> = emptyMap(),
) {
    /** Best effort: a read-only models dir must not break `runa ls`. */
    fun save(
        fileSystem: FileSystem,
        path: Path,
    ) {
        try {
            fileSystem.writeAtomically(path) { writeUtf8(format.encodeToString(serializer(), this@ModelIndex)) }
        } catch (_: IOException) {
        }
    }

    companion object {
        const val FILE_NAME: String = ".index.json"

        // Raise it when the meaning of a field changes, so older caches are rebuilt.
        private const val VERSION = 1
        private val format = Json { ignoreUnknownKeys = true }

        fun load(
            fileSystem: FileSystem,
            path: Path,
        ): ModelIndex =
            try {
                format
                    .decodeFromString(serializer(), fileSystem.read(path) { readUtf8() })
                    .takeIf { it.version == VERSION } ?: ModelIndex()
            } catch (_: IOException) {
                ModelIndex()
            } catch (_: SerializationException) {
                ModelIndex()
            } catch (_: IllegalArgumentException) {
                ModelIndex()
            }
    }
}
