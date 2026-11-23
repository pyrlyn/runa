package runa.daemon

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import okio.Path
import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.core.UserInputError
import runa.core.api.AutoChoice
import runa.core.api.InstanceDescription
import runa.core.api.MemoryEstimateDescription
import runa.engine.Engine
import runa.engine.LoadedModel
import runa.engine.MemoryEstimate
import runa.store.StoreListing
import runa.store.StoredModel
import runa.store.find
import runa.store.isNamed
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicReference
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.concurrent.atomics.decrementAndFetch
import kotlin.concurrent.atomics.incrementAndFetch
import kotlin.time.ComparableTimeMark
import kotlin.time.Duration
import kotlin.time.TimeSource

/** What to load and how; a null field takes the registry's [LoadDefaults]. */
public data class LoadSpec(
    /** A model of the store, named as `runa ls` shows it. */
    val model: String,
    val contextLength: Int? = null,
    val gpu: GpuOffload? = null,
    /** How many requests may use the instance at once. */
    val parallel: Int? = null,
    /** Idle time after which the instance is unloaded; zero for never. */
    val ttl: Duration? = null,
    val identifier: String? = null,
    /**
     * Choose [gpu] and [contextLength] by the machine's memory, so that the model fits. Either one
     * that is given is kept; the [LoadDefaults] do not count as given.
     */
    val auto: Boolean = false,
)

/** The `context_length`, `gpu` and `ttl` settings of `config.json`. */
public data class LoadDefaults(
    val contextLength: Int? = null,
    val gpu: GpuOffload = GpuOffload.Max,
    val ttl: Duration? = null,
)

/** No instance and no stored model has the name a request gave. */
public class UnknownModelError(
    message: String,
) : RuntimeException(message)

public class LoadResult(
    public val instance: InstanceDescription,
    public val estimate: MemoryEstimateDescription,
    public val loadTime: Duration,
    /** What `auto` chose; null when the load did not ask for it. */
    public val auto: AutoChoice? = null,
)

/** What a load would take, and what `auto` would choose for it when asked. */
public class LoadEstimate(
    public val memory: MemoryEstimateDescription,
    public val auto: AutoChoice?,
)

/**
 * The models the daemon holds in memory. An instance is one loaded copy of a model, known by its
 * identifier; a model can be loaded more than once. Safe to call from any coroutine.
 *
 * @param models reads the model store; called on every load, so a model downloaded while the
 * daemon runs can be loaded.
 * @param modelsDir where [models] looks, for error messages.
 * @param scope runs the idle timers; cancelling it stops them but unloads nothing, [close] does.
 */
@OptIn(ExperimentalAtomicApi::class)
public class ModelRegistry(
    private val engine: Engine,
    private val models: () -> StoreListing,
    private val modelsDir: Path,
    private val scope: CoroutineScope,
    private val defaults: LoadDefaults = LoadDefaults(),
    /** The memory a load with `auto` plans with; asked on every such load. */
    private val memory: () -> MachineMemory = { MachineMemory(systemBytes = null) },
    private val time: TimeSource.WithComparableMarks = TimeSource.Monotonic,
    /** Told about every model that was loaded, whoever asked for it. */
    private val onLoaded: (LoadResult) -> Unit = {},
    /** Told the identifier of an instance that was unloaded for being idle. */
    private val onEvicted: (String) -> Unit = {},
) {
    private val lock = Mutex()
    private val instances = LinkedHashMap<String, Instance>()

    // Identifiers of loads in flight, so two loads cannot both settle on the same one.
    private val reserved = HashSet<String>()
    private var closed = false

    // Held while a request loads the model it names, so the requests behind it wait for that load.
    private val loadingOnDemand = Mutex()

    /** What [spec] would take, without loading anything. */
    public suspend fun estimate(spec: LoadSpec): LoadEstimate {
        val model = find(spec.model)
        val config = config(spec, model)
        return LoadEstimate(describe(engine.estimateMemory(model.path, config)), choice(spec, config))
    }

    /**
     * Loads a model and returns once it is ready for requests.
     *
     * @throws UserInputError when the model is unknown, the identifier is taken or a value is invalid.
     */
    public suspend fun load(
        spec: LoadSpec,
        onProgress: (Float) -> Unit = {},
    ): LoadResult {
        val model = find(spec.model)
        val config = config(spec, model)
        val estimate = engine.estimateMemory(model.path, config)
        val parallel = spec.parallel ?: 1
        if (parallel < 1) throw UserInputError("parallel must be at least 1")
        val ttl = (spec.ttl ?: defaults.ttl)?.takeIf { it > Duration.ZERO }
        val id = lock.withLock { reserve(spec.identifier, model) }
        val started = time.markNow()
        val instance =
            try {
                val loaded = engine.load(model.path, config, onProgress)
                Instance(id, model, loaded, config, estimate.totalBytes, parallel, ttl, time.markNow())
            } catch (e: Throwable) {
                withContext(NonCancellable) { lock.withLock { reserved -= id } }
                throw e
            }
        val accepted =
            lock.withLock {
                reserved -= id
                if (!closed) instances[id] = instance
                !closed
            }
        if (!accepted) {
            instance.release()
            throw IllegalStateException("The daemon is shutting down")
        }
        if (ttl != null) instance.timer = scope.launch { unloadWhenIdle(instance, ttl) }
        return LoadResult(instance.describe(), describe(estimate), started.elapsedNow(), choice(spec, config)).also(onLoaded)
    }

    /** Runs [block] with the model of instance [id], holding one of its parallel slots. */
    public suspend fun <T> use(
        id: String,
        block: suspend (LoadedModel) -> T,
    ): T {
        val instance = lock.withLock { instances[id] } ?: throw unknown(id)
        return instance.slots.withPermit {
            // It may have been unloaded while this request waited for a slot.
            if (instance.released) throw unknown(id)
            instance.active.incrementAndFetch()
            try {
                block(instance.model)
            } finally {
                instance.idleSince.store(time.markNow())
                instance.active.decrementAndFetch()
            }
        }
    }

    /**
     * Runs [block] with the model that a request calls [name]: the instance with that identifier,
     * else a loaded instance of the stored model of that name, else that model, loaded now with the
     * defaults. [block] also gets the identifier of the instance it runs on.
     *
     * @throws UnknownModelError when [name] is neither an instance nor a stored model.
     * @throws UserInputError when [name] fits more than one stored model.
     */
    public suspend fun <T> useNamed(
        name: String,
        block: suspend (id: String, model: LoadedModel) -> T,
    ): T {
        val id = resolve(name)
        return use(id) { block(id, it) }
    }

    private suspend fun resolve(name: String): String {
        if (lock.withLock { name in instances }) return name
        val listing = models()
        val model =
            when (listing.models.count { it.isNamed(name) }) {
                0 -> throw UnknownModelError("No model is called '$name'. `runa ls` lists the models, `runa ps` the loaded ones")
                else -> listing.find(name, modelsDir)
            }

        suspend fun loaded() = lock.withLock { instances.values.firstOrNull { it.stored.key == model.key }?.id }
        return loaded() ?: loadingOnDemand.withLock { loaded() ?: load(LoadSpec(model.key.toString())).instance.id }
    }

    /**
     * Unloads one instance after the requests using it have finished.
     *
     * @throws UserInputError when no instance has this identifier.
     */
    public suspend fun unload(id: String) {
        val instance = lock.withLock { instances.remove(id) } ?: throw unknown(id)
        instance.timer?.cancel()
        instance.release()
    }

    /** Unloads everything and returns the identifiers. */
    public suspend fun unloadAll(): List<String> {
        val all =
            lock.withLock {
                instances.values.toList().also { instances.clear() }
            }
        all.forEach {
            it.timer?.cancel()
            it.release()
        }
        return all.map { it.id }
    }

    public suspend fun instances(): List<InstanceDescription> = lock.withLock { instances.values.map { it.describe() } }

    /** Unloads everything; later loads fail. */
    public suspend fun close() {
        lock.withLock { closed = true }
        unloadAll()
    }

    private suspend fun unloadWhenIdle(
        instance: Instance,
        ttl: Duration,
    ) {
        while (true) {
            val idle = instance.idleSince.load().elapsedNow()
            val busy = instance.active.load() > 0
            if (!busy && idle >= ttl) break
            delay(if (busy) ttl else ttl - idle)
        }
        // Only if it is still the registered one: an explicit unload may have won the race.
        val mine = lock.withLock { instances[instance.id] === instance && instances.remove(instance.id) != null }
        // The timer must not be cancelled halfway through freeing the model.
        if (mine) {
            withContext(NonCancellable) { instance.release() }
            onEvicted(instance.id)
        }
    }

    private fun find(name: String): StoredModel = models().find(name, modelsDir)

    private suspend fun config(
        spec: LoadSpec,
        model: StoredModel,
    ): LoadConfig {
        // With auto the configured defaults step aside: it is asked to choose what the request left open.
        val context = spec.contextLength ?: if (spec.auto) null else defaults.contextLength ?: LoadConfig.DEFAULT_CONTEXT_LENGTH
        if (context != null && context < 1) throw UserInputError("context_length must be at least 1")
        if (!spec.auto) return LoadConfig(contextLength = context, gpu = spec.gpu ?: defaults.gpu)
        val held = lock.withLock { instances.values.map { HeldModel(it.estimatedBytes, it.config.gpu) } }
        return chooseLoadConfig(
            model = model.key.toString(),
            memory = memory(),
            held = held,
            contextLength = context,
            gpu = spec.gpu,
            trainedContext = model.contextLength?.coerceAtMost(Int.MAX_VALUE.toLong())?.toInt(),
        ) { context -> engine.estimateMemory(model.path, LoadConfig(contextLength = context)).totalBytes }
    }

    private fun choice(
        spec: LoadSpec,
        config: LoadConfig,
    ): AutoChoice? = config.contextLength?.takeIf { spec.auto }?.let { AutoChoice(it, config.gpu.id) }

    private fun describe(estimate: MemoryEstimate) =
        MemoryEstimateDescription(estimate.weightsBytes, estimate.contextBytes, estimate.totalBytes)

    /** Call with [lock] held. Without a wish the model's key is taken, then `key:2`, `key:3`, and so on. */
    private fun reserve(
        wanted: String?,
        model: StoredModel,
    ): String {
        fun taken(id: String) = id in instances || id in reserved
        val id =
            if (wanted != null) {
                if (!IDENTIFIER.matches(wanted)) {
                    throw UserInputError("'$wanted' cannot be an identifier: use up to $MAX_IDENTIFIER letters, digits and . _ - : @ /")
                }
                if (taken(wanted)) throw UserInputError("A model is already loaded as '$wanted'; unload it or choose another identifier")
                wanted
            } else {
                val key = model.key.toString()
                generateSequence(1) { it + 1 }.map { if (it == 1) key else "$key:$it" }.first { !taken(it) }
            }
        reserved += id
        return id
    }

    private fun unknown(id: String) = UserInputError("No loaded model is called '$id'. `runa ps` lists the loaded models")

    private fun Instance.describe() =
        InstanceDescription(
            id = id,
            model = stored.key.toString(),
            contextLength = model.contextLength,
            gpu = config.gpu.id,
            parallel = parallel,
            ttlSeconds = ttl?.inWholeSeconds,
            idleSeconds = if (active.load() > 0) 0 else idleSince.load().elapsedNow().inWholeSeconds,
            sizeBytes = stored.sizeBytes,
        )

    private class Instance(
        val id: String,
        val stored: StoredModel,
        val model: LoadedModel,
        val config: LoadConfig,
        /** What the engine estimated it takes, for the loads that plan around it. */
        val estimatedBytes: Long,
        val parallel: Int,
        val ttl: Duration?,
        loadedAt: ComparableTimeMark,
    ) {
        val slots = Semaphore(parallel)
        val active = AtomicInt(0)
        val idleSince = AtomicReference(loadedAt)
        var timer: Job? = null

        @kotlin.concurrent.Volatile
        var released = false
            private set

        /** Waits for the requests in flight, then frees the model. The slots are never given back. */
        suspend fun release() =
            withContext(NonCancellable) {
                repeat(parallel) { slots.acquire() }
                released = true
                model.close()
                repeat(parallel) { slots.release() }
            }
    }

    private companion object {
        const val MAX_IDENTIFIER = 128
        val IDENTIFIER = Regex("[A-Za-z0-9._:@/-]{1,$MAX_IDENTIFIER}")
    }
}
