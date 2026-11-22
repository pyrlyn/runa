package runa.daemon

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.currentTime
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.core.UserInputError
import runa.core.api.AutoChoice
import runa.engine.testing.FakeEngine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

private const val GIB = 1024L * 1024 * 1024

@OptIn(ExperimentalCoroutinesApi::class)
class ModelRegistryTest {
    private val engine = FakeEngine()
    private val evicted = mutableListOf<String>()

    // Virtual time: the idle timers and the registry's clock both follow the test scheduler.
    private fun TestScope.registry(
        defaults: LoadDefaults = LoadDefaults(),
        memory: MachineMemory = MachineMemory(systemBytes = null),
        engine: FakeEngine = this@ModelRegistryTest.engine,
    ) = ModelRegistry(engine, { listing }, modelsDir, backgroundScope, defaults, { memory }, testScheduler.timeSource) { evicted += it }

    private suspend fun ModelRegistry.ids() = instances().map { it.id }

    @Test
    fun aModelCanBeLoadedMoreThanOnce() =
        runTest {
            val registry = registry()
            assertEquals(ALPHA, registry.load(LoadSpec("alpha")).instance.id)
            assertEquals("$ALPHA:2", registry.load(LoadSpec("acme/alpha")).instance.id)
            assertEquals("mine", registry.load(LoadSpec(ALPHA, identifier = "mine")).instance.id)
            assertEquals(listOf(ALPHA, "$ALPHA:2", "mine"), registry.ids())
            assertEquals(3, engine.loads.size)
            assertTrue(registry.instances().all { it.model == ALPHA })
        }

    @Test
    fun aRequestLoadsTheModelItNamesOnceWithTheDefaults() =
        runTest {
            val registry = registry(LoadDefaults(contextLength = 256, gpu = GpuOffload.Off, ttl = 60.seconds))
            val used = mutableListOf<String>()
            repeat(3) { launch { registry.useNamed("acme/alpha") { id, model -> used += "$id ${model.contextLength}" } } }
            runCurrent()
            assertEquals(List(3) { "$ALPHA 256" }, used)
            assertEquals(LoadConfig(contextLength = 256, gpu = GpuOffload.Off), engine.loads.single().config)
            // Loaded on demand, it goes the way of every idle model.
            advanceTimeBy(61.seconds)
            assertEquals(listOf(ALPHA), evicted)
        }

    @Test
    fun aRequestMeansAnInstanceBeforeItMeansAStoredModel() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("beta", identifier = "alpha"))
            registry.useNamed("alpha") { id, _ -> assertEquals("alpha", id) }
            // The stored model beta is loaded already, under another name.
            registry.useNamed("acme/beta") { id, _ -> assertEquals("alpha", id) }
            assertEquals(1, engine.loads.size)
            registry.useNamed(ALPHA) { id, _ -> assertEquals(ALPHA, id) }
            assertEquals(2, engine.loads.size)
            assertFailsWith<UnknownModelError> { registry.useNamed("gamma") { _, _ -> } }
        }

    @Test
    fun aFreedIdentifierIsUsedAgain() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha"))
            registry.load(LoadSpec("alpha"))
            registry.unload(ALPHA)
            assertEquals(ALPHA, registry.load(LoadSpec("alpha")).instance.id)
        }

    @Test
    fun anIdentifierInUseIsRefused() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha", identifier = "mine"))
            val error = assertFailsWith<UserInputError> { registry.load(LoadSpec("beta", identifier = "mine")) }
            assertTrue("mine" in error.message.orEmpty())
            assertFailsWith<UserInputError> { registry.load(LoadSpec("beta", identifier = "two words")) }
            assertFailsWith<UserInputError> { registry.load(LoadSpec("beta", identifier = "")) }
            assertEquals(1, engine.loads.size)
        }

    @Test
    fun whatCannotBeLoadedIsRefusedBeforeTheEngineSeesIt() =
        runTest {
            val registry = registry()
            assertFailsWith<UserInputError> { registry.load(LoadSpec("gamma")) }
            assertFailsWith<UserInputError> { registry.load(LoadSpec("alpha", parallel = 0)) }
            assertFailsWith<UserInputError> { registry.load(LoadSpec("alpha", contextLength = 0)) }
            assertEquals(0, engine.loads.size)
            assertEquals(emptyList(), registry.ids())
            // The refused loads must not keep their identifier.
            assertEquals(ALPHA, registry.load(LoadSpec("alpha")).instance.id)
        }

    @Test
    fun theRequestWinsOverTheConfiguredDefaults() =
        runTest {
            registry().load(LoadSpec("alpha"))
            assertEquals(LoadConfig(LoadConfig.DEFAULT_CONTEXT_LENGTH, GpuOffload.Max), engine.loads.last().config)

            val configured = registry(LoadDefaults(contextLength = 1024, gpu = GpuOffload.Off, ttl = 60.seconds))
            val byDefault = configured.load(LoadSpec("alpha")).instance
            assertEquals(LoadConfig(1024, GpuOffload.Off), engine.loads.last().config)
            assertEquals(60, byDefault.ttlSeconds)
            assertEquals("off", byDefault.gpu)
            assertEquals(1, byDefault.parallel)

            val asked =
                configured
                    .load(
                        LoadSpec("alpha", contextLength = 512, gpu = GpuOffload.Ratio(0.5), parallel = 3, ttl = Duration.ZERO),
                    ).instance
            assertEquals(LoadConfig(512, GpuOffload.Ratio(0.5)), engine.loads.last().config)
            assertEquals(512, asked.contextLength)
            assertEquals(3, asked.parallel)
            // Zero is how a request says "never" when the configuration has a ttl.
            assertEquals(null, asked.ttlSeconds)
        }

    @Test
    fun anEstimateLoadsNothing() =
        runTest {
            val registry = registry()
            val estimate = registry.estimate(LoadSpec("alpha", contextLength = 100)).memory
            assertEquals(1_000, estimate.weightsBytes)
            assertEquals(200, estimate.contextBytes)
            assertEquals(1_200, estimate.totalBytes)
            assertEquals(0, engine.loads.size)
            assertFailsWith<UserInputError> { registry.estimate(LoadSpec("gamma")) }
        }

    @Test
    fun unloadFreesTheModel() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha"))
            registry.load(LoadSpec("beta"))
            registry.load(LoadSpec("beta"))
            registry.unload(ALPHA)
            assertTrue(engine.loads[0].model.closed)
            assertFalse(engine.loads[1].model.closed)
            assertFailsWith<UserInputError> { registry.unload(ALPHA) }
            assertFailsWith<UserInputError> { registry.use(ALPHA) {} }

            assertEquals(listOf("acme/beta@Q4_K_M", "acme/beta@Q4_K_M:2"), registry.unloadAll())
            assertTrue(engine.loads.all { it.model.closed })
            assertEquals(emptyList(), registry.ids())
        }

    @Test
    fun anIdleModelIsUnloadedAfterItsTtl() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha", ttl = 2.seconds))
            registry.load(LoadSpec("beta"))
            advanceTimeBy(1_900)
            assertEquals(1, registry.instances().first().idleSeconds)
            assertEquals(2, registry.ids().size)
            advanceTimeBy(200)
            assertEquals(listOf("acme/beta@Q4_K_M"), registry.ids())
            assertTrue(engine.loads[0].model.closed)
            assertEquals(listOf(ALPHA), evicted)
            // Without a ttl a model stays for good.
            advanceTimeBy(1_000_000)
            assertEquals(listOf("acme/beta@Q4_K_M"), registry.ids())
        }

    @Test
    fun aRequestStartsTheTtlAgain() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha", ttl = 2.seconds))
            advanceTimeBy(1_500)
            registry.use(ALPHA) {}
            advanceTimeBy(1_900)
            assertEquals(listOf(ALPHA), registry.ids())
            advanceTimeBy(200)
            assertEquals(emptyList(), registry.ids())
        }

    @Test
    fun aBusyModelIsNotUnloaded() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha", ttl = 2.seconds))
            val request = launch { registry.use(ALPHA) { delay(10.seconds) } }
            advanceTimeBy(9_000)
            assertEquals(0, registry.instances().single().idleSeconds)
            advanceTimeBy(2_900)
            // 1.9 s after the request ended.
            assertTrue(request.isCompleted)
            assertEquals(listOf(ALPHA), registry.ids())
            advanceTimeBy(200)
            assertEquals(emptyList(), registry.ids())
        }

    @Test
    fun unloadWaitsForTheRequestsInFlight() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha"))
            val model = engine.loads.single().model
            var closedDuringRequest = true
            launch {
                registry.use(ALPHA) {
                    delay(5.seconds)
                    closedDuringRequest = model.closed
                }
            }
            runCurrent()
            val unload = launch { registry.unload(ALPHA) }
            advanceTimeBy(4_000)
            assertFalse(unload.isCompleted)
            assertFalse(model.closed)
            // Already gone from the list, so nothing new starts on it.
            assertEquals(emptyList(), registry.ids())
            advanceUntilIdle()
            assertTrue(unload.isCompleted)
            assertFalse(closedDuringRequest)
            assertTrue(model.closed)
        }

    @Test
    fun parallelIsTheNumberOfRequestsAtOnce() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha", identifier = "one"))
            registry.load(LoadSpec("alpha", identifier = "three", parallel = 3))

            suspend fun timeOfThreeRequests(id: String): Long {
                val start = currentTime
                val requests = List(3) { launch { registry.use(id) { delay(1.seconds) } } }
                requests.forEach { it.join() }
                return currentTime - start
            }
            assertEquals(3_000, timeOfThreeRequests("one"))
            assertEquals(1_000, timeOfThreeRequests("three"))
        }

    @Test
    fun autoChoosesWhatTheRequestLeftOpenAndSaysWhat() =
        runTest {
            val gpu = FakeEngine(gpuOffload = true)
            // The configured defaults are not a request: auto chooses in their place.
            val registry =
                registry(
                    LoadDefaults(contextLength = 512, gpu = GpuOffload.Off),
                    MachineMemory(16 * GIB, GpuMemory(12 * GIB, unified = true)),
                    gpu,
                )

            val loaded = registry.load(LoadSpec("alpha", auto = true))
            // The stored model was trained with 2048 tokens, and everything fits this machine.
            assertEquals(LoadConfig(2048, GpuOffload.Max), gpu.loads.single().config)
            assertEquals(AutoChoice(2048, "max"), loaded.auto)

            val asked = registry.estimate(LoadSpec("alpha", contextLength = 1000, gpu = GpuOffload.Off, auto = true))
            assertEquals(AutoChoice(1000, "off"), asked.auto)
            assertEquals(3_000, asked.memory.totalBytes)
            assertEquals(1, gpu.loads.size)

            assertEquals(null, registry.load(LoadSpec("beta")).auto)
            assertEquals(null, registry.estimate(LoadSpec("beta")).auto)
            assertFailsWith<UserInputError> { registry.estimate(LoadSpec("alpha", contextLength = 0, auto = true)) }
        }

    @Test
    fun autoCountsWhatIsLoadedAlready() =
        runTest {
            val engine = FakeEngine()
            engine.weightsBytes = 5 * GIB
            // 16 GiB less the fifth that stays free: room for two of these models, not for three.
            val registry = registry(memory = MachineMemory(16 * GIB), engine = engine)

            repeat(2) { assertEquals(AutoChoice(2048, "off"), registry.load(LoadSpec("alpha", auto = true)).auto) }
            val refused = assertFailsWith<UserInputError> { registry.load(LoadSpec("alpha", auto = true)) }
            assertEquals(
                "auto finds no way to fit $ALPHA: with a context of 2048 it takes about 5368 MB, " +
                    "and this machine has 3006 MB of memory for it. Give gpu and context_length to load it anyway",
                refused.message,
            )
            // Who names the settings loads it all the same.
            registry.load(LoadSpec("alpha", contextLength = 2048, gpu = GpuOffload.Off))
            assertEquals(3, registry.ids().size)
            // And what is unloaded makes room again: with two of the three gone there is space for one more.
            registry.unload(ALPHA)
            assertFailsWith<UserInputError> { registry.load(LoadSpec("alpha", auto = true)) }
            registry.unload("$ALPHA:2")
            registry.load(LoadSpec("alpha", auto = true))
        }

    @Test
    fun autoNeedsToKnowTheMemory() =
        runTest {
            val refused = assertFailsWith<UserInputError> { registry().load(LoadSpec("alpha", auto = true)) }
            assertTrue("does not tell" in refused.message.orEmpty(), refused.message)
            assertEquals(0, engine.loads.size)
        }

    @Test
    fun aClosedRegistryUnloadsEverythingAndLoadsNothing() =
        runTest {
            val registry = registry()
            registry.load(LoadSpec("alpha", ttl = 2.seconds))
            registry.close()
            assertTrue(
                engine.loads
                    .single()
                    .model.closed,
            )
            assertFailsWith<IllegalStateException> { registry.load(LoadSpec("alpha")) }
            assertTrue(
                engine.loads
                    .last()
                    .model.closed,
            )
            assertEquals(emptyList(), registry.ids())
        }
}
