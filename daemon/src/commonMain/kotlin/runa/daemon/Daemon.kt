package runa.daemon

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import okio.FileSystem
import runa.config.DaemonInfo
import runa.config.DaemonSettings
import runa.config.DiscoveryFile
import runa.core.ExitCode
import runa.core.LogLevel
import runa.engine.Engine
import runa.platform.FileLock
import runa.platform.catchTermination
import runa.platform.currentProcessId
import runa.platform.terminationRequested
import runa.store.ModelStore
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * The daemon process: holds the loaded models and serves the API on a loopback port until it is
 * told to stop by SIGTERM or `POST /api/v1/shutdown`. `runa server start` makes it serve the same
 * API on a second, public address as well. Returns the process exit code.
 *
 * @param engine creates the inference engine; it gets the sink for the engine's own log.
 */
@OptIn(ExperimentalAtomicApi::class)
public fun runDaemon(
    settings: DaemonSettings,
    engine: (log: (LogLevel, String) -> Unit) -> Engine,
    fileSystem: FileSystem = FileSystem.SYSTEM,
): Int {
    val log = DaemonLog(settings.logLevel)
    val paths = settings.paths
    fileSystem.createDirectories(checkNotNull(paths.daemonLockFile.parent))
    // Held until the process ends, however it ends: the OS drops the lock of a dead process.
    val lock = FileLock.tryAcquire(paths.daemonLockFile)
    if (lock == null) {
        log.error("Another runa daemon already runs for ${paths.stateDir}")
        return ExitCode.BACKEND
    }
    return lock.use {
        // Whatever an earlier daemon left behind describes a process that no longer exists.
        DiscoveryFile(fileSystem, paths.discoveryFile).delete()
        runBlocking { serve(settings, engine(log::at), fileSystem, log) }
        ExitCode.OK
    }
}

@OptIn(ExperimentalAtomicApi::class)
private suspend fun serve(
    settings: DaemonSettings,
    engine: Engine,
    fileSystem: FileSystem,
    log: DaemonLog,
) {
    val pid = currentProcessId()
    val store = ModelStore(fileSystem, settings.modelsDir)
    val timers = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val registry =
        ModelRegistry(
            engine = engine,
            models = store::list,
            modelsDir = settings.modelsDir,
            scope = timers,
            defaults = LoadDefaults(settings.contextLength, settings.gpu, settings.ttl),
            onLoaded = { log.info("Loaded ${it.instance.model} as '${it.instance.id}' in ${it.loadTime.inWholeMilliseconds} ms") },
            onEvicted = { log.info("Unloaded '$it': idle for longer than its ttl") },
        )
    val stopRequested = AtomicBoolean(false)
    var port = 0
    val access = Access(randomHex() + randomHex())
    val failures = Failures(log)
    // Each server that starts brings Ktor's signal handlers back, see below.
    val public = PublicServer(access, log, afterStart = ::catchTermination)
    val api =
        DaemonApi(
            registry = registry,
            engine = engine,
            store = store::list,
            access = access,
            failures = failures,
            openAi = OpenAiApi(registry, store::list, access, failures, log),
            server = public,
            version = settings.version,
            pid = pid,
            port = { port },
            started = TimeSource.Monotonic.markNow(),
            log = log,
            requestShutdown = { stopRequested.store(true) },
        )
    val discovery = DiscoveryFile(fileSystem, settings.paths.discoveryFile)
    var server: HttpServer? = null
    try {
        server = listen(LOOPBACK, settings.port ?: 0, log) { api.install(this, Listener.PRIVATE) }
        // After the server has started, because Ktor installs signal handlers of its own then: they
        // would stop the server and leave this process running without one.
        catchTermination()
        port = server.port()
        // Written only now: whoever finds the file can connect at once.
        discovery.write(DaemonInfo(pid, port, access.token, settings.version))
        log.info("runa ${settings.version} (pid $pid) listens on $LOOPBACK:$port with ${engine.info.name} ${engine.info.version}")
        // Signal handlers can only set a flag, so the flag is polled.
        while (!stopRequested.load() && !terminationRequested()) delay(POLL)
        log.info("Stopping")
    } finally {
        discovery.delete(pid)
        public.stop()
        server?.stopSuspend(gracePeriodMillis = GRACE_MILLIS, timeoutMillis = STOP_TIMEOUT_MILLIS)
        registry.close()
        timers.cancel()
        log.info("Stopped")
    }
}

/** 122 random bits from the operating system's generator, which is what `Uuid.random` reads; the token is two of these. */
@OptIn(ExperimentalUuidApi::class)
internal fun randomHex(): String = Uuid.random().toHexString()

private const val LOOPBACK = "127.0.0.1"
private val POLL = 100.milliseconds
private const val GRACE_MILLIS = 200L
private const val STOP_TIMEOUT_MILLIS = 2_000L
