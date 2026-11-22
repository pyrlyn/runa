package runa.daemon

import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.receiveText
import io.ktor.server.response.respondBytesWriter
import io.ktor.server.response.respondText
import io.ktor.server.routing.RoutingContext
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import io.ktor.utils.io.ByteWriteChannel
import io.ktor.utils.io.writeStringUtf8
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.serialization.DeserializationStrategy
import kotlinx.serialization.SerializationException
import kotlinx.serialization.SerializationStrategy
import runa.core.GpuOffload
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.core.api.DaemonStatus
import runa.core.api.EngineDescription
import runa.core.api.LoadProgress
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.ModelDescription
import runa.core.api.ModelList
import runa.core.api.ShutdownResponse
import runa.core.api.UnloadRequest
import runa.core.api.UnloadResponse
import runa.engine.Engine
import runa.store.StoreListing
import kotlin.time.Duration.Companion.seconds
import kotlin.time.DurationUnit
import kotlin.time.TimeMark

/**
 * The daemon's native HTTP API (PLAN.md, section 6). Every route needs the bearer [token], except
 * the status probe: it tells a client that the process it found is a runa daemon, and nothing else.
 */
internal class DaemonApi(
    private val registry: ModelRegistry,
    private val engine: Engine,
    private val store: () -> StoreListing,
    val token: String,
    private val version: String,
    private val pid: Int,
    /** Known only once the server listens. */
    private val port: () -> Int,
    private val started: TimeMark,
    private val log: DaemonLog,
    private val requestShutdown: () -> Unit,
) {
    fun install(application: Application) {
        application.routing {
            get(DaemonRoutes.STATUS) { answer { call.respond(DaemonStatus.serializer(), status()) } }
            get(DaemonRoutes.MODELS) { guarded { call.respond(ModelList.serializer(), models()) } }
            post(DaemonRoutes.LOAD) { guarded { load(call, call.receive(LoadRequest.serializer())) } }
            post(
                DaemonRoutes.UNLOAD,
            ) { guarded { call.respond(UnloadResponse.serializer(), unload(call.receive(UnloadRequest.serializer()))) } }
            post(DaemonRoutes.SHUTDOWN) {
                guarded {
                    call.respond(ShutdownResponse.serializer(), ShutdownResponse("stopping"))
                    log.info("Shutdown requested over the API")
                    requestShutdown()
                }
            }
        }
    }

    private suspend fun status() =
        DaemonStatus(
            version = version,
            pid = pid,
            port = port(),
            uptimeSeconds = started.elapsedNow().inWholeSeconds,
            engine = engine.info.let { EngineDescription(it.name, it.version, it.accelerators) },
            loadedInstances = registry.instances().size,
        )

    private suspend fun models(): ModelList {
        val loaded = registry.instances().groupBy { it.model }
        return ModelList(
            store().models.map { model ->
                val key = model.key.toString()
                ModelDescription(
                    key = key,
                    type = model.kind.id,
                    displayName = model.name,
                    architecture = model.architecture,
                    quantization = model.quantization,
                    params = model.parameterCount,
                    sizeBytes = model.sizeBytes,
                    maxContextLength = model.contextLength,
                    loadedInstances = loaded[key].orEmpty(),
                )
            },
        )
    }

    private suspend fun load(
        call: ApplicationCall,
        request: LoadRequest,
    ) {
        val spec = spec(request)
        when {
            request.estimateOnly -> {
                call.respond(LoadResponse.serializer(), LoadResponse(LoadResponse.ESTIMATED, registry.estimate(spec)))
            }

            request.stream -> {
                // Whatever is wrong with the request is found before the stream starts, so it still gets a 400.
                registry.estimate(spec)
                call.respondBytesWriter(ContentType.Text.EventStream) { streamLoad(spec) }
            }

            else -> {
                call.respond(LoadResponse.serializer(), loaded(registry.load(spec)))
            }
        }
    }

    /** `progress` events while the engine reads the file, then one `loaded` or `error`. */
    private suspend fun ByteWriteChannel.streamLoad(spec: LoadSpec) {
        suspend fun <T> event(
            name: String,
            serializer: SerializationStrategy<T>,
            value: T,
        ) {
            writeStringUtf8("event: $name\ndata: ${DaemonJson.encodeToString(serializer, value)}\n\n")
            flush()
        }
        // A client that goes away fails the next write, which cancels the load with this scope.
        coroutineScope {
            // The engine reports from its own thread and must never wait for a slow client.
            val progress = Channel<Float>(Channel.CONFLATED)
            val load =
                async {
                    try {
                        Result.success(registry.load(spec) { progress.trySend(it) })
                    } catch (e: CancellationException) {
                        throw e
                    } catch (e: Exception) {
                        Result.failure(e)
                    } finally {
                        progress.close()
                    }
                }
            for (value in progress) event(DaemonRoutes.EVENT_PROGRESS, LoadProgress.serializer(), LoadProgress(value))
            load.await().fold(
                onSuccess = { event(DaemonRoutes.EVENT_LOADED, LoadResponse.serializer(), loaded(it)) },
                onFailure = { event(DaemonRoutes.EVENT_ERROR, ApiError.serializer(), failure(it).second) },
            )
        }
    }

    private fun loaded(result: LoadResult): LoadResponse {
        log.info("Loaded ${result.instance.model} as '${result.instance.id}' in ${result.loadTime.inWholeMilliseconds} ms")
        return LoadResponse(LoadResponse.LOADED, result.estimate, result.instance, result.loadTime.toDouble(DurationUnit.SECONDS))
    }

    private suspend fun unload(request: UnloadRequest): UnloadResponse {
        val id = request.instanceId
        val unloaded =
            when {
                request.all -> registry.unloadAll()
                id != null -> listOf(id).onEach { registry.unload(it) }
                else -> throw UserInputError("Give instance_id, or all: true")
            }
        unloaded.forEach { log.info("Unloaded '$it'") }
        return UnloadResponse(unloaded)
    }

    private fun spec(request: LoadRequest): LoadSpec {
        val ttl = request.ttl
        if (ttl != null && ttl < 0) throw UserInputError("ttl must not be negative")
        return LoadSpec(
            model = request.model,
            contextLength = request.contextLength,
            gpu = request.gpu?.let { GpuOffload.parse(it) ?: throw UserInputError("gpu must be off, max, or a number from 0 to 1") },
            parallel = request.parallel,
            ttl = ttl?.seconds,
            identifier = request.identifier,
        )
    }

    private suspend fun RoutingContext.guarded(block: suspend () -> Unit) =
        answer {
            if (!authorized(call.request.headers[HttpHeaders.Authorization])) throw Unauthorized()
            block()
        }

    /** Turns whatever [block] throws into the API's error body. */
    private suspend fun RoutingContext.answer(block: suspend () -> Unit) {
        try {
            block()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            val (status, body) = failure(e)
            call.respond(ApiError.serializer(), body, status)
        }
    }

    private fun failure(error: Throwable): Pair<HttpStatusCode, ApiError> {
        fun body(
            message: String,
            type: String,
        ) = ApiError(ApiError.Detail(message, type))
        return when (error) {
            is Unauthorized -> {
                HttpStatusCode.Unauthorized to body("A valid bearer token is required", ApiError.UNAUTHORIZED)
            }

            is UserInputError -> {
                HttpStatusCode.BadRequest to body(error.message.orEmpty(), ApiError.INVALID_REQUEST)
            }

            is SerializationException -> {
                HttpStatusCode.BadRequest to body("The request body is not valid: ${error.message}", ApiError.INVALID_REQUEST)
            }

            else -> {
                log.error("Request failed: ${error.stackTraceToString()}")
                HttpStatusCode.InternalServerError to body(error.message ?: error.toString(), ApiError.SERVER_ERROR)
            }
        }
    }

    private fun authorized(header: String?): Boolean {
        val given = header?.takeIf { it.startsWith(BEARER, ignoreCase = true) }?.substring(BEARER.length) ?: return false
        val a = given.encodeToByteArray()
        val b = token.encodeToByteArray()
        // Every byte is compared, so the time taken says nothing about where the tokens differ.
        var difference = a.size xor b.size
        for (i in a.indices) difference = difference or (a[i].toInt() xor b[i % b.size].toInt())
        return difference == 0
    }

    private suspend fun <T> ApplicationCall.respond(
        serializer: SerializationStrategy<T>,
        value: T,
        status: HttpStatusCode = HttpStatusCode.OK,
    ) = respondText(DaemonJson.encodeToString(serializer, value), ContentType.Application.Json, status)

    private suspend fun <T> ApplicationCall.receive(serializer: DeserializationStrategy<T>): T =
        DaemonJson.decodeFromString(serializer, receiveText().ifBlank { "{}" })

    private class Unauthorized : RuntimeException()

    private companion object {
        const val BEARER = "Bearer "
    }
}
