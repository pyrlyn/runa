package runa.daemon

import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.application.ApplicationCallPipeline
import io.ktor.server.application.call
import io.ktor.server.request.httpMethod
import io.ktor.server.request.path
import io.ktor.server.response.header
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.merge
import kotlinx.coroutines.flow.onSubscription
import kotlinx.serialization.DeserializationStrategy
import kotlinx.serialization.SerializationStrategy
import okio.Path
import runa.core.GpuOffload
import runa.core.UserInputError
import runa.core.api.ApiError
import runa.core.api.ChatRequest
import runa.core.api.DaemonJson
import runa.core.api.DaemonRoutes
import runa.core.api.DaemonStatus
import runa.core.api.EngineDescription
import runa.core.api.LoadProgress
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.LogEvent
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
import kotlin.time.TimeSource

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
    private val chat: ChatApi,
    private val server: PublicServer,
    private val version: String,
    private val pid: Int,
    /** Known only once the server listens. */
    private val port: () -> Int,
    private val started: TimeMark,
    private val log: DaemonLog,
    /** Where the log goes when it goes to a file. */
    private val logFile: Path? = null,
    private val requestShutdown: () -> Unit,
) {
    fun install(
        application: Application,
        listener: Listener,
    ) {
        // Every request is an event of the log, whatever becomes of it.
        application.intercept(ApplicationCallPipeline.Monitoring) {
            val asked = TimeSource.Monotonic.markNow()
            try {
                proceed()
            } finally {
                // A request no route took is answered after this pipeline, with a 404.
                val status = call.response.status() ?: HttpStatusCode.NotFound
                val request = call.request
                log.request("${request.httpMethod.value} ${request.path()} ${status.value} ${asked.elapsedNow().inWholeMilliseconds} ms")
            }
        }
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
            post(DaemonRoutes.CHAT) { guarded { chat.answer(call, call.receive(ChatRequest.serializer())) } }
            get(DaemonRoutes.LOGS) { guarded { streamLog(call) } }
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
            engine = engine.info.let { EngineDescription(it.name, it.version, it.accelerators, it.gpuOffload) },
            loadedInstances = registry.instances().size,
            server = server.describe(),
            logFile = logFile?.toString(),
        )

    /**
     * The log from now on, until the client goes away or the daemon stops. The `listening` event
     * says that the events after it are all there: whatever the client does next, it will see.
     */
    private suspend fun streamLog(call: ApplicationCall) {
        val source = choice(call, DaemonRoutes.LOGS_SOURCE, LogEvent.SOURCES)
        val filter = choice(call, DaemonRoutes.LOGS_FILTER, LogEvent.FILTERS)
        val events =
            log.events
                .onSubscription { emit(LISTENING) }
                .filter { it === LISTENING || ((source == null || it.source == source) && (filter == null || it.type == filter)) }
                .map {
                    if (it === LISTENING) {
                        sseFrame("{}", DaemonRoutes.EVENT_LISTENING)
                    } else {
                        sseFrame(DaemonJson.encodeToString(LogEvent.serializer(), it), DaemonRoutes.EVENT_LOG)
                    }
                }
        // A comment, which a client skips. It keeps a quiet stream from looking dead to whatever
        // sits in between, and it is the write that fails once the client has gone away.
        val keepAlive =
            flow {
                while (true) {
                    delay(KEEP_ALIVE)
                    emit(": keep-alive\n\n")
                }
            }
        call.response.header(HttpHeaders.CacheControl, "no-cache")
        call.respondBytesWriter(ContentType.Text.EventStream) {
            merge(events, keepAlive).collect {
                writeStringUtf8(it)
                flush()
            }
        }
    }

    private fun choice(
        call: ApplicationCall,
        parameter: String,
        choices: List<String>,
    ): String? {
        val value = call.request.queryParameters[parameter] ?: return null
        if (value !in choices) throw UserInputError("$parameter must be one of: ${choices.joinToString(", ")}")
        return value
    }

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

    private companion object {
        val KEEP_ALIVE = 15.seconds

        /** Not an event of the log: it marks, in the flow of them, the moment from which none is missed. */
        val LISTENING = LogEvent(time = "", source = "", type = "", text = "")
    }
}
