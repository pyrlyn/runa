package runa.cli

import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.toByteArray
import io.ktor.client.request.HttpRequestData
import io.ktor.client.request.HttpResponseData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okio.IOException

/**
 * What the CLI sees of a running LM Studio: its v1 REST API with two models, in the JSON of its
 * documentation, and as strict as the real one about a key it does not know. The client's own
 * contract with the documented examples is tested in `:client`.
 */
internal class FakeLmStudio(
    private val port: Int = 1234,
    /** What a download ends as. */
    private val downloadEnds: String = "completed",
) {
    /** `POST /api/v1/models/load {...}`, in order. */
    val requests = mutableListOf<String>()

    /** The `Authorization` header of every request. */
    val tokens = mutableListOf<String?>()

    /** Instance to context length. */
    val loaded = mutableMapOf<String, Int>()
    private var answers = 0
    private var statusChecks = 0

    val http: HttpClientEngine = MockEngine { request -> answer(request) }

    private suspend fun MockRequestHandleScope.answer(request: HttpRequestData): HttpResponseData {
        if (request.url.port != port) throw IOException("Connection refused")
        val text = request.body.toByteArray().decodeToString()
        val path = request.url.encodedPath
        requests += "${request.method.value} $path $text".trim()
        tokens += request.headers[HttpHeaders.Authorization]
        val body = if (text.isEmpty()) JsonObject(emptyMap()) else Json.parseToJsonElement(text).jsonObject
        return when {
            path == "/api/v1/models" -> json("""{"models":[${MODELS.joinToString(",") { model(it) }}]}""")
            path == "/api/v1/models/load" -> load(body)
            path == "/api/v1/models/unload" -> unload(body.string("instance_id"))
            path == "/api/v1/chat" -> chat(body)
            path == "/api/v1/models/download" -> download(body.string("model"))
            path.startsWith("/api/v1/models/download/status/") -> downloadStatus(path.substringAfterLast('/'))
            else -> json("""{"error":"Unexpected endpoint or method. (${request.method.value} $path)"}""", HttpStatusCode.NotFound)
        }
    }

    private fun model(key: String): String {
        val instances =
            loaded.filterKeys { it == key }.entries.joinToString(",") { (id, context) ->
                val parallel = if (key == TINY) ""","parallel":4""" else ""
                """{"id":"$id","config":{"context_length":$context$parallel,"flash_attention":true}}"""
            }
        return if (key == TINY) {
            """{"type":"llm","publisher":"acme","key":"$TINY","display_name":"Tiny","architecture":"llama",""" +
                """"quantization":{"name":"Q4_K_M","bits_per_weight":4},"size_bytes":668000000,"params_string":"1.1B",""" +
                """"loaded_instances":[$instances],"max_context_length":2048,"format":"gguf"}"""
        } else {
            """{"type":"embedding","publisher":"acme","key":"$EMBEDDING","display_name":"Small",""" +
                """"quantization":{"name":"F16","bits_per_weight":16},"size_bytes":84106624,"params_string":null,""" +
                """"loaded_instances":[$instances],"max_context_length":512,"format":"gguf"}"""
        }
    }

    private fun MockRequestHandleScope.load(body: JsonObject): HttpResponseData {
        val unknown = body.keys - setOf("model", "context_length")
        if (unknown.isNotEmpty()) {
            val keys = unknown.joinToString(", ") { "'$it'" }
            return error("invalid_request", "Unrecognized key(s) in object: $keys", HttpStatusCode.BadRequest)
        }
        val key = body.string("model")
        if (key !in MODELS) return error("model_not_found", "Model $key not found in downloaded models", HttpStatusCode.NotFound)
        loaded[key] = body["context_length"]?.jsonPrimitive?.content?.toInt() ?: 2048
        return json("""{"type":"llm","instance_id":"$key","load_time_seconds":1.5,"status":"loaded"}""")
    }

    private fun MockRequestHandleScope.unload(id: String): HttpResponseData {
        if (loaded.remove(id) == null) {
            return error("model_not_found", "Model with instance identifier '$id' is not loaded.", HttpStatusCode.NotFound)
        }
        return json("""{"instance_id":"$id"}""")
    }

    /**
     * Thinks, then answers with what it was asked and what the request carried besides, behind the
     * two empty lines a reasoning model leaves in front of its answer.
     */
    private fun MockRequestHandleScope.chat(body: JsonObject): HttpResponseData {
        check(body["stream"] == JsonPrimitive(true)) { "the CLI always streams" }
        val instance = body.string("model")
        if (instance !in loaded) return error("invalid_request", "Model $instance is not loaded", HttpStatusCode.NotFound)
        val system = body["system_prompt"]?.jsonPrimitive?.content.orEmpty()
        val head = "\n\n" + (if (system.isEmpty()) "" else "[$system] ") + "echo:"
        val tail =
            " ${body.string("input")}" +
                body["previous_response_id"]
                    ?.jsonPrimitive
                    ?.content
                    ?.let { " (after $it)" }
                    .orEmpty()
        val id = if (body["store"] == JsonPrimitive(false)) "" else ""","response_id":"resp_${++answers}""""
        val result =
            """{"model_instance_id":"$instance","output":[{"type":"reasoning","content":"Let me think."},""" +
                """{"type":"message","content":${JsonPrimitive(head + tail)}}],"stats":{"input_tokens":9,"total_output_tokens":12,""" +
                """"reasoning_output_tokens":7,"tokens_per_second":29.753900615398926,"time_to_first_token_seconds":0.212}$id}"""
        val events =
            listOf(
                """{"type":"chat.start","model_instance_id":"$instance"}""",
                """{"type":"prompt_processing.start"}""",
                """{"type":"prompt_processing.end"}""",
                """{"type":"reasoning.start"}""",
                """{"type":"reasoning.delta","content":"Let me think."}""",
                """{"type":"reasoning.end"}""",
                """{"type":"message.start"}""",
                """{"type":"message.delta","content":${JsonPrimitive(head)}}""",
                """{"type":"message.delta","content":${JsonPrimitive(tail)}}""",
                """{"type":"message.end"}""",
                """{"type":"chat.end","result":$result}""",
            )
        val stream = events.joinToString("") { "event: ${Json.parseToJsonElement(it).jsonObject.string("type")}\ndata: $it\n\n" }
        return respond(stream, HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "text/event-stream"))
    }

    private fun MockRequestHandleScope.download(model: String): HttpResponseData =
        if (model == TINY) {
            json("""{"status":"already_downloaded"}""")
        } else {
            json("""{"job_id":"$JOB","status":"downloading","total_size_bytes":1000,"started_at":"$TIME"}""")
        }

    /** Half done at the first question, over at the second. */
    private fun MockRequestHandleScope.downloadStatus(job: String): HttpResponseData {
        if (job != JOB) return error("job_not_found", "Download job $job not found", HttpStatusCode.NotFound)
        val start = """"job_id":"$JOB","total_size_bytes":1000,"started_at":"$TIME""""
        return when {
            ++statusChecks == 1 -> json("""{$start,"status":"downloading","downloaded_bytes":500,"bytes_per_second":250}""")
            downloadEnds == "completed" -> json("""{$start,"status":"completed","downloaded_bytes":1000,"completed_at":"$TIME"}""")
            else -> json("""{$start,"status":"$downloadEnds","downloaded_bytes":500}""")
        }
    }

    private fun JsonObject.string(key: String): String = this[key]?.jsonPrimitive?.content.orEmpty()

    private fun MockRequestHandleScope.error(
        type: String,
        message: String,
        status: HttpStatusCode,
    ) = json("""{"error":{"type":"$type","message":${JsonPrimitive(message)}}}""", status)

    private fun MockRequestHandleScope.json(
        text: String,
        status: HttpStatusCode = HttpStatusCode.OK,
    ) = respond(text, status, headersOf(HttpHeaders.ContentType, "application/json"))

    companion object {
        const val TINY = "acme/tiny"
        const val EMBEDDING = "text-embedding-small"
        const val JOB = "job_1"
        private const val TIME = "2026-01-02T03:04:05.000Z"
        private val MODELS = listOf(TINY, EMBEDDING)
    }
}
