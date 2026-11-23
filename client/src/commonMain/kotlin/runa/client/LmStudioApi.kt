package runa.client

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

// The JSON of LM Studio's v1 REST API (https://lmstudio.ai/docs/developer/rest, read 2026-11-22
// for LM Studio 0.4.x), as far as runa uses it. Its chat is the one of runa's own API.

/**
 * How LM Studio is written to and read: it refuses a request with a key it does not know and may
 * add fields to an answer at any time, so a null is left out and an unknown field is skipped.
 */
internal val LmStudioJson: Json = Json { ignoreUnknownKeys = true }

/** `GET /api/v1/models`. */
@Serializable
internal data class LmModels(
    val models: List<LmModel>,
)

@Serializable
internal data class LmModel(
    /** `llm`, `embedding` or `decision`. */
    val type: String,
    val key: String,
    @SerialName("display_name") val displayName: String? = null,
    val architecture: String? = null,
    val quantization: LmQuantization? = null,
    @SerialName("size_bytes") val sizeBytes: Long,
    /** A label such as `7B`, not always a plain count: `26B-A4B`. */
    @SerialName("params_string") val paramsString: String? = null,
    @SerialName("loaded_instances") val loadedInstances: List<LmInstance> = emptyList(),
    @SerialName("max_context_length") val maxContextLength: Long? = null,
)

@Serializable
internal data class LmQuantization(
    val name: String? = null,
)

@Serializable
internal data class LmInstance(
    val id: String,
    val config: LmLoadConfig,
)

@Serializable
internal data class LmLoadConfig(
    @SerialName("context_length") val contextLength: Int,
    /** Absent for embedding models. */
    val parallel: Int? = null,
)

/** `POST /api/v1/models/load`. */
@Serializable
internal data class LmLoadRequest(
    val model: String,
    @SerialName("context_length") val contextLength: Int? = null,
)

@Serializable
internal data class LmLoaded(
    @SerialName("instance_id") val instanceId: String,
    @SerialName("load_time_seconds") val loadTimeSeconds: Double? = null,
)

/** `POST /api/v1/models/unload`, and its answer. */
@Serializable
internal data class LmUnload(
    @SerialName("instance_id") val instanceId: String,
)

/** `POST /api/v1/models/download`. */
@Serializable
internal data class LmDownloadRequest(
    val model: String,
    /** Only for a Hugging Face link. */
    val quantization: String? = null,
)

/**
 * A download job of LM Studio: the answer of `POST /api/v1/models/download` and of
 * `GET /api/v1/models/download/status/<job>`. Only [status] is always there.
 */
@Serializable
public data class LmStudioDownload(
    /** Absent when the model was already downloaded. */
    @SerialName("job_id") val jobId: String? = null,
    /** [DOWNLOADING], [PAUSED], [COMPLETED], [FAILED] or [ALREADY_DOWNLOADED]. */
    val status: String,
    @SerialName("total_size_bytes") val totalSizeBytes: Long? = null,
    @SerialName("downloaded_bytes") val downloadedBytes: Long? = null,
    @SerialName("bytes_per_second") val bytesPerSecond: Double? = null,
    @SerialName("started_at") val startedAt: String? = null,
    @SerialName("completed_at") val completedAt: String? = null,
) {
    public companion object {
        public const val DOWNLOADING: String = "downloading"
        public const val PAUSED: String = "paused"
        public const val COMPLETED: String = "completed"
        public const val FAILED: String = "failed"
        public const val ALREADY_DOWNLOADED: String = "already_downloaded"
    }
}
