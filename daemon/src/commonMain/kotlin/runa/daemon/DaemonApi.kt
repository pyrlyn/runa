package runa.daemon

import io.ktor.http.ContentType
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respondBytesWriter
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
import runa.core.api.ServerDescription
import runa.core.api.ServerRequest
import runa.core.api.ShutdownResponse
import runa.core.api.UnloadRequest
import runa.core.api.UnloadResponse
import runa.engine.Engine
import runa.store.StoreListing
import kotlin.time.Duration.Companion.seconds
import kotlin.time.DurationUnit
import kotlin.time.TimeMark

/**
 * The daemon's native HTTP API (PLAN.md, section 6), with the OpenAI-compatible one beside it.
 * Every native route needs the daemon's token, except the status probe on the private listener:
 * it tells a client that the process it found is a runa daemon, and nothing else.
 */
internal class DaemonApi(
    private val registry: ModelRegistry,
    private val engine: Engine,
    private val store: () -> StoreListing,
    private val access: Access,
    private val failures: Failures,
    private val openAi: OpenAiApi,
    private val server: PublicServer,
    private val version: String,
    private val pid: Int,
    /** Known only once the server listens. */
    private val port: () -> Int,
    private val started: TimeMark,
    private val log: DaemonLog,
    private val requestShutdown: () -> Unit,
) {
    fun install(
        application: Application,
        listener: Listener,
    ) {
        application.routing {
            openAi.install(this, listener)
            get(DaemonRoutes.STATUS) {
                // Open only where nothing but this machine can ask.
                if (listener == Listener.PRIVATE) {
                    failures.answer(call) { call.respond(DaemonStatus.serializer(), status()) }
                } else {
                    guarded { call.respond(DaemonStatus.serializer(), status()) }
                }
            }
            get(DaemonRoutes.MODELS) { guarded { call.respond(ModelList.serializer(), models()) } }
            post(DaemonRoutes.LOAD) { guarded { load(call, call.receive(LoadRequest.serializer())) } }
            post(
                DaemonRoutes.UNLOAD,
            ) { guarded { call.respond(UnloadResponse.serializer(), unload(call.receive(UnloadRequest.serializer()))) } }
            post(DaemonRoutes.SERVER) {
                guarded { call.respond(ServerDescription.serializer(), serve(call.receive(ServerRequest.serializer()), listener)) }
            }
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
            server = server.describe(),
        )

    private suspend fun serve(
        request: ServerRequest,
        listener: Listener,
    ): ServerDescription {
        if (request.running) return server.start(request) { install(this, Listener.PUBLIC) }
        // Closing it would cut off the very request that asks for it.
        if (listener == Listener.PUBLIC) throw UserInputError("The server cannot be stopped through itself; ask the daemon's own port")
        server.stop()
        return server.describe()
    }

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
            writeStringUtf8(sseFrame(DaemonJson.encodeToString(serializer, value), name))
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
                onFailure = { event(DaemonRoutes.EVENT_ERROR, ApiError.serializer(), failures.describe(it).second) },
            )
        }
    }

    private fun loaded(result: LoadResult) =
        LoadResponse(LoadResponse.LOADED, result.estimate, result.instance, result.loadTime.toDouble(DurationUnit.SECONDS))

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
        failures.answer(call) {
            access.requireToken(call)
            block()
        }

    private suspend fun <T> ApplicationCall.respond(
        serializer: SerializationStrategy<T>,
        value: T,
    ) = respondJson(DaemonJson, serializer, value)

    private suspend fun <T> ApplicationCall.receive(serializer: DeserializationStrategy<T>): T = receiveJson(DaemonJson, serializer)
}
