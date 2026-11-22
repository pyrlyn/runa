package runa.core.api

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

// The JSON of the daemon's native API (PLAN.md, section 6). The daemon writes these types and the
// client reads them, so the two cannot drift apart; `--json` output of `ps` and `status` is the same.

/**
 * How both sides write and read the API: every field is written, defaults and nulls too, and a
 * field this build does not know is skipped, so an older client can talk to a newer daemon.
 */
public val DaemonJson: Json =
    Json {
        encodeDefaults = true
        ignoreUnknownKeys = true
    }

/** `GET /api/v1/status`. */
@Serializable
public data class DaemonStatus(
    val version: String,
    val pid: Int,
    val port: Int,
    @SerialName("uptime_seconds") val uptimeSeconds: Long,
    val engine: EngineDescription,
    @SerialName("loaded_instances") val loadedInstances: Int,
)

@Serializable
public data class EngineDescription(
    val name: String,
    val version: String,
    val accelerators: List<String>,
)

/** `GET /api/v1/models`. */
@Serializable
public data class ModelList(
    val models: List<ModelDescription>,
)

@Serializable
public data class ModelDescription(
    /** `publisher/repo@variant`, as `runa ls` shows it. */
    val key: String,
    /** `llm` or `embedding`. */
    val type: String,
    @SerialName("display_name") val displayName: String?,
    val architecture: String?,
    val quantization: String?,
    val params: Long,
    @SerialName("size_bytes") val sizeBytes: Long,
    @SerialName("max_context_length") val maxContextLength: Long?,
    @SerialName("loaded_instances") val loadedInstances: List<InstanceDescription>,
)

/** One loaded copy of a model. */
@Serializable
public data class InstanceDescription(
    val id: String,
    /** The key of the model it was loaded from. */
    val model: String,
    @SerialName("context_length") val contextLength: Int,
    /** `off`, `max` or a ratio, as asked for at load time. */
    val gpu: String,
    val parallel: Int,
    /** Seconds of idleness after which it is unloaded; null for never. */
    @SerialName("ttl_seconds") val ttlSeconds: Long?,
    @SerialName("idle_seconds") val idleSeconds: Long,
    @SerialName("size_bytes") val sizeBytes: Long,
)

/** `POST /api/v1/models/load`. A null field takes the daemon's default. */
@Serializable
public data class LoadRequest(
    val model: String,
    @SerialName("context_length") val contextLength: Int? = null,
    val gpu: String? = null,
    val parallel: Int? = null,
    /** Seconds; 0 for never. */
    val ttl: Long? = null,
    val identifier: String? = null,
    /** Answer with the memory estimate and load nothing. */
    @SerialName("estimate_only") val estimateOnly: Boolean = false,
    /** Answer with server-sent events: `progress` while loading, then `loaded` or `error`. */
    val stream: Boolean = false,
)

@Serializable
public data class LoadResponse(
    /** [LOADED] or [ESTIMATED]. */
    val status: String,
    val estimate: MemoryEstimateDescription,
    /** Null when only an estimate was asked for. */
    val instance: InstanceDescription? = null,
    @SerialName("load_time_seconds") val loadTimeSeconds: Double? = null,
) {
    public companion object {
        public const val LOADED: String = "loaded"
        public const val ESTIMATED: String = "estimated"
    }
}

@Serializable
public data class MemoryEstimateDescription(
    @SerialName("weights_bytes") val weightsBytes: Long,
    @SerialName("context_bytes") val contextBytes: Long,
    @SerialName("total_bytes") val totalBytes: Long,
)

/** The data of a `progress` event; [progress] goes from 0 to 1. */
@Serializable
public data class LoadProgress(
    val progress: Float,
)

/** `POST /api/v1/models/unload`: one instance, or all of them. */
@Serializable
public data class UnloadRequest(
    @SerialName("instance_id") val instanceId: String? = null,
    val all: Boolean = false,
)

@Serializable
public data class UnloadResponse(
    val unloaded: List<String>,
)

/** `POST /api/v1/shutdown`. */
@Serializable
public data class ShutdownResponse(
    val status: String,
)

/** The body of every response that is not a success, in the shape the OpenAI API uses. */
@Serializable
public data class ApiError(
    val error: Detail,
) {
    @Serializable
    public data class Detail(
        val message: String,
        /** [INVALID_REQUEST], [UNAUTHORIZED] or [SERVER_ERROR]. */
        val type: String,
    )

    public companion object {
        public const val INVALID_REQUEST: String = "invalid_request_error"
        public const val UNAUTHORIZED: String = "authentication_error"
        public const val SERVER_ERROR: String = "server_error"
    }
}

/** The paths and event names of the API, shared by the daemon and its client. */
public object DaemonRoutes {
    public const val STATUS: String = "/api/v1/status"
    public const val MODELS: String = "/api/v1/models"
    public const val LOAD: String = "/api/v1/models/load"
    public const val UNLOAD: String = "/api/v1/models/unload"
    public const val SHUTDOWN: String = "/api/v1/shutdown"

    public const val EVENT_PROGRESS: String = "progress"
    public const val EVENT_LOADED: String = "loaded"
    public const val EVENT_ERROR: String = "error"
}
