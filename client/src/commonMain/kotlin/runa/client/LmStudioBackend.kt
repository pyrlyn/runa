package runa.client

import io.ktor.client.engine.HttpClientEngine
import io.ktor.http.HttpStatusCode
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.DaemonRoutes
import runa.core.api.InstanceDescription
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.ModelDescription
import runa.core.api.ModelList
import runa.core.api.UnloadRequest
import runa.core.api.UnloadResponse

/**
 * Talks to a running LM Studio (0.4.0 or newer) over its documented v1 REST API, which runa's own
 * API was shaped after: the paths are the same, the chat is the same, and the rest is translated
 * here. LM Studio is never started, and nothing but its public REST API is used.
 *
 * @param engine null for the platform's own HTTP engine; tests pass their own.
 * @param baseUrl for example `http://127.0.0.1:1234`.
 * @param token an API token of LM Studio, when it asks for one; null sends none.
 */
public class LmStudioBackend(
    engine: HttpClientEngine? = null,
    private val baseUrl: String,
    token: String?,
) : ModelBackend {
    private val api =
        object : ApiTransport(engine, baseUrl, token, LmStudioJson) {
            override val service = "LM Studio"
            override val unreadable = "$baseUrl sent an answer that runa cannot read; is it LM Studio 0.4.0 or newer?"
            override val chatCutShort = "LM Studio closed the connection before the answer was complete"

            override fun failure(
                status: HttpStatusCode,
                body: String,
            ): RuntimeException {
                val error = errorOf(body)
                return when {
                    status == HttpStatusCode.Unauthorized || status == HttpStatusCode.Forbidden -> {
                        BackendError("LM Studio at $baseUrl rejected the API token; give the right one with --token or $TOKEN_ENV")
                    }

                    // Whatever the status: a model LM Studio does not have is a 404, and still the user's to put right.
                    error?.type in USER_ERRORS -> {
                        UserInputError(checkNotNull(error).message)
                    }

                    status == HttpStatusCode.BadRequest -> {
                        UserInputError(error?.message ?: "LM Studio rejected the request")
                    }

                    else -> {
                        BackendError("LM Studio at $baseUrl answered ${status.value}" + error?.let { ": ${it.message}" }.orEmpty())
                    }
                }
            }

            override fun failure(error: ApiError.Detail): RuntimeException =
                if (error.type in USER_ERRORS) UserInputError(error.message) else BackendError("The answer was cut short: ${error.message}")
        }

    override suspend fun models(): ModelList = ModelList(api.get(DaemonRoutes.MODELS, LmModels.serializer()).models.map { it.describe() })

    /**
     * Loads a model with the context length the request names, or LM Studio's own. LM Studio reports
     * no progress, and its REST API has nothing for the other settings of a [LoadRequest].
     *
     * @throws UserInputError when the request asks for one of them.
     */
    override suspend fun load(
        request: LoadRequest,
        onProgress: (Float) -> Unit,
    ): LoadResponse {
        val refused =
            listOfNotNull(
                "gpu".takeIf { request.gpu != null },
                "parallel".takeIf { request.parallel != null },
                "ttl".takeIf { request.ttl != null },
                "identifier".takeIf { request.identifier != null },
                "estimate_only".takeIf { request.estimateOnly },
            )
        if (refused.isNotEmpty()) {
            throw UserInputError(
                "LM Studio's REST API loads a model with its context length and nothing else: it cannot take ${refused.joinToString(", ")}",
            )
        }
        val loaded =
            api.post(
                DaemonRoutes.LOAD,
                LmLoadRequest.serializer(),
                LmLoadRequest(keyOf(request.model), request.contextLength),
                LmLoaded.serializer(),
            )
        // The answer names the instance; what it was loaded with is in the list.
        val instance =
            models().models.flatMap { it.loadedInstances }.firstOrNull { it.id == loaded.instanceId }
                ?: throw BackendError("LM Studio loaded '${loaded.instanceId}' and does not list it")
        return LoadResponse(LoadResponse.LOADED, estimate = null, instance, loaded.loadTimeSeconds)
    }

    /**
     * LM Studio loads a model by its whole key, and answers a shorter name with an internal error.
     * runa's own models are also named by the part after the publisher, so that is accepted here
     * too; a name that fits no model, or several, is left for LM Studio to refuse.
     */
    private suspend fun keyOf(name: String): String {
        val keys = api.get(DaemonRoutes.MODELS, LmModels.serializer()).models.map { it.key }
        if (name in keys) return name
        return keys.singleOrNull { it.substringAfterLast('/').equals(name, ignoreCase = true) } ?: name
    }

    override suspend fun chat(
        request: ChatRequest,
        onText: (String) -> Unit,
    ): ChatResponse = api.chat(request, onText)

    /** LM Studio unloads one instance per request, so "all" is every instance it lists. */
    override suspend fun unload(request: UnloadRequest): UnloadResponse {
        val instances =
            if (request.all) {
                models().models.flatMap { model ->
                    model.loadedInstances.map { it.id }
                }
            } else {
                listOfNotNull(request.instanceId)
            }
        return UnloadResponse(
            instances.map {
                api.post(DaemonRoutes.UNLOAD, LmUnload.serializer(), LmUnload(it), LmUnload.serializer()).instanceId
            },
        )
    }

    /**
     * Asks LM Studio to download [model]: a name from its catalog, or a Hugging Face link, for which
     * [quantization] may choose the variant. Returns at once; [downloadStatus] follows the job.
     */
    public suspend fun download(
        model: String,
        quantization: String? = null,
    ): LmStudioDownload =
        api.post(DOWNLOAD, LmDownloadRequest.serializer(), LmDownloadRequest(model, quantization), LmStudioDownload.serializer())

    public suspend fun downloadStatus(jobId: String): LmStudioDownload = api.get("$DOWNLOAD_STATUS/$jobId", LmStudioDownload.serializer())

    override fun close(): Unit = api.close()

    private fun LmModel.describe(): ModelDescription =
        ModelDescription(
            key = key,
            type = type,
            displayName = displayName,
            architecture = architecture,
            quantization = quantization?.name,
            params = parameterCount(paramsString),
            sizeBytes = sizeBytes,
            maxContextLength = maxContextLength,
            loadedInstances =
                loadedInstances.map {
                    InstanceDescription(
                        id = it.id,
                        model = key,
                        contextLength = it.config.contextLength,
                        gpu = null,
                        parallel = it.config.parallel,
                        ttlSeconds = null,
                        idleSeconds = null,
                        // The size of the files: LM Studio does not say what an instance takes in memory.
                        sizeBytes = sizeBytes,
                    )
                },
        )

    /** `{"error": {"type", "message", ...}}`, or `{"error": "..."}` for a path LM Studio does not have. */
    private fun errorOf(body: String): ApiError.Detail? =
        try {
            when (val error = (LmStudioJson.parseToJsonElement(body) as? JsonObject)?.get("error")) {
                is JsonObject -> LmStudioJson.decodeFromJsonElement(ApiError.Detail.serializer(), error)
                is JsonPrimitive -> ApiError.Detail(error.content, type = "unknown")
                else -> null
            }
        } catch (_: SerializationException) {
            null
        }

    public companion object {
        /** Where LM Studio's server listens unless it is told otherwise. */
        public const val DEFAULT_PORT: Int = 1234

        /** The environment variable LM Studio's own examples read the API token from. */
        public const val TOKEN_ENV: String = "LM_API_TOKEN"

        private const val DOWNLOAD = "/api/v1/models/download"
        private const val DOWNLOAD_STATUS = "$DOWNLOAD/status"

        // The `type` of the errors that say the request, not LM Studio, is wrong.
        private val USER_ERRORS = setOf("invalid_request", "model_not_found", "job_not_found")
        private val PARAMETER_COUNT = Regex("(\\d+(?:\\.\\d+)?)([KMBT])", RegexOption.IGNORE_CASE)
        private val SCALES = mapOf('K' to 1e3, 'M' to 1e6, 'B' to 1e9, 'T' to 1e12)

        /** `7B` as 7 000 000 000; null for a label that is not one count, such as `26B-A4B`. */
        internal fun parameterCount(label: String?): Long? {
            val (number, scale) = PARAMETER_COUNT.matchEntire(label.orEmpty())?.destructured ?: return null
            return (number.toDouble() * SCALES.getValue(scale.single().uppercaseChar())).toLong()
        }
    }
}
