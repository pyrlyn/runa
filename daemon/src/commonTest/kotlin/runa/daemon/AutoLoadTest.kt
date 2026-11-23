package runa.daemon

import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.core.UserInputError
import runa.engine.EngineInfo
import runa.platform.Cpu
import runa.platform.Gpu
import runa.platform.GpuSource
import runa.platform.HardwareSurvey
import runa.platform.HostOs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

private const val GIB = 1024L * 1024 * 1024
private const val MIB = 1024L * 1024

/** A model whose weights take [weights] bytes and whose context takes [perToken] bytes a token. */
private class Model(
    val weights: Long,
    val perToken: Long,
    val trained: Int? = 131_072,
) {
    fun bytes(context: Int): Long = weights + perToken * context
}

private val mac = MachineMemory(36 * GIB, GpuMemory(27 * GIB, unified = true))
private val card = MachineMemory(32 * GIB, GpuMemory(8 * GIB, unified = false))
private val cpuOnly = MachineMemory(16 * GIB)

private fun choose(
    memory: MachineMemory,
    model: Model,
    held: List<HeldModel> = emptyList(),
    contextLength: Int? = null,
    gpu: GpuOffload? = null,
): LoadConfig = chooseLoadConfig("acme/model@Q4", memory, held, contextLength, gpu, model.trained, model::bytes)

class AutoLoadTest {
    @Test
    fun aModelThatFitsTheGpuGoesThereWithTheLongestContext() {
        // 5 GiB of weights and 128 KiB a token: 32768 tokens are 4 GiB more.
        val model = Model(5 * GIB, 128 * 1024)
        assertEquals(LoadConfig(32_768, GpuOffload.Max), choose(mac, model))
        // No longer than the model was trained with, and never longer than the cap.
        assertEquals(LoadConfig(8192, GpuOffload.Max), choose(mac, Model(5 * GIB, 128 * 1024, trained = 8192)))
        assertEquals(LoadConfig(1000, GpuOffload.Max), choose(mac, Model(5 * GIB, 128 * 1024, trained = 1000)))
        assertEquals(LoadConfig(32_768, GpuOffload.Max), choose(mac, Model(5 * GIB, 128 * 1024, trained = null)))
        assertEquals(LoadConfig(32_768, GpuOffload.Max), choose(mac, Model(5 * GIB, 128 * 1024, trained = 1_000_000)))
    }

    @Test
    fun theContextShrinksBeforeAnythingLeavesTheGpu() {
        // 7.2 GiB of the card are planned with: 5 GiB of weights leave room for 16384 tokens, not for 32768.
        assertEquals(LoadConfig(16_384, GpuOffload.Max), choose(card, Model(5 * GIB, 128 * 1024)))
        // On a Mac the GPU's share is the machine's memory: 20 GiB of weights leave room for 32768 tokens of 128 KiB, not of 512 KiB.
        assertEquals(LoadConfig(32_768, GpuOffload.Max), choose(mac, Model(20 * GIB, 128 * 1024)))
        assertEquals(LoadConfig(8192, GpuOffload.Max), choose(mac, Model(20 * GIB, 512 * 1024)))
    }

    @Test
    fun aModelTooLargeForTheCardIsSplitWithAModestContext() {
        // 12 GiB of weights and 4096 tokens of 128 KiB are 12.5 GiB, of which 7.2 GiB fit the card.
        val config = choose(card, Model(12 * GIB, 128 * 1024))
        assertEquals(LoadConfig(4096, GpuOffload.Ratio(0.57)), config)
        // With next to nothing for the card it is not worth it.
        val small = MachineMemory(32 * GIB, GpuMemory(1 * GIB, unified = false))
        assertEquals(LoadConfig(32_768, GpuOffload.Off), choose(small, Model(12 * GIB, 128 * 1024)))
    }

    @Test
    fun withoutAGpuTheMachinesMemoryDecides() {
        // 16 GiB less a fifth: 12.8 GiB.
        assertEquals(LoadConfig(32_768, GpuOffload.Off), choose(cpuOnly, Model(5 * GIB, 128 * 1024)))
        assertEquals(LoadConfig(16_384, GpuOffload.Off), choose(cpuOnly, Model(10 * GIB, 128 * 1024)))
        // A small machine keeps 2 GiB, which is more than its fifth.
        assertEquals(LoadConfig(2048, GpuOffload.Off), choose(MachineMemory(4 * GIB), Model(2 * GIB - 2048 * MIB / 1024, MIB / 1024)))
        assertFailsWith<UserInputError> { choose(MachineMemory(4 * GIB), Model(2 * GIB, MIB / 1024)) }
    }

    @Test
    fun whatTheRequestFixedIsKept() {
        val model = Model(5 * GIB, 128 * 1024)
        assertEquals(LoadConfig(32_768, GpuOffload.Off), choose(mac, model, gpu = GpuOffload.Off))
        // Half of the 9 GiB that 32768 tokens make fits the card, and the other half the machine.
        assertEquals(LoadConfig(32_768, GpuOffload.Ratio(0.5)), choose(card, model, gpu = GpuOffload.Ratio(0.5)))
        // A longer context than auto would take is the request's to ask for.
        assertEquals(LoadConfig(65_536, GpuOffload.Max), choose(mac, model, contextLength = 65_536))
        assertEquals(LoadConfig(65_536, GpuOffload.Ratio(0.55)), choose(card, model, contextLength = 65_536))
        assertEquals(LoadConfig(100, GpuOffload.Max), choose(card, model, contextLength = 100, gpu = GpuOffload.Max))
        // Fixed and too large: nothing is left to choose.
        assertFailsWith<UserInputError> { choose(card, model, contextLength = 65_536, gpu = GpuOffload.Max) }
    }

    @Test
    fun theLoadedModelsTakeTheirPlace() {
        val model = Model(5 * GIB, 128 * 1024)
        // 4 GiB of the card are taken: 3.2 GiB are left, and the model is split.
        assertEquals(LoadConfig(4096, GpuOffload.Ratio(0.58)), choose(card, model, held = listOf(HeldModel(4 * GIB, GpuOffload.Max))))
        // What runs on the CPU does not take from the card.
        assertEquals(LoadConfig(16_384, GpuOffload.Max), choose(card, model, held = listOf(HeldModel(4 * GIB, GpuOffload.Off))))
        // On a Mac there is one memory, wherever a model computes.
        assertEquals(LoadConfig(8192, GpuOffload.Max), choose(mac, model, held = listOf(HeldModel(21 * GIB, GpuOffload.Off))))
        // Without a GPU the share a loaded model asked for means nothing.
        assertEquals(LoadConfig(16_384, GpuOffload.Off), choose(cpuOnly, model, held = listOf(HeldModel(5 * GIB, GpuOffload.Max))))
    }

    @Test
    fun whatFitsNowhereIsRefusedWithTheNumbers() {
        val refused = assertFailsWith<UserInputError> { choose(card, Model(60 * GIB, 128 * 1024)) }
        assertEquals(
            "auto finds no way to fit acme/model@Q4: with a context of 2048 it takes about 64692 MB, and this machine has " +
                "27487 MB of memory and 7730 MB on the GPU for it. Give gpu and context_length to load it anyway",
            refused.message,
        )
        assertTrue(
            "38654 MB of memory for it" in
                assertFailsWith<UserInputError> { choose(MachineMemory(45 * GIB), Model(60 * GIB, 0)) }.message.orEmpty(),
        )
        val unknown = assertFailsWith<UserInputError> { choose(MachineMemory(systemBytes = null), Model(GIB, 0)) }
        assertTrue("does not tell" in unknown.message.orEmpty(), unknown.message)
    }

    /** The acceptance of the phase: whatever is chosen, its estimate fits the memory the machine reported. */
    @Test
    fun whatIsChosenAlwaysFitsTheReportedMemory() {
        val machines =
            listOf(4L, 8, 16, 36, 64, 128).flatMap { ram ->
                listOf(MachineMemory(ram * GIB)) +
                    listOf(4L, 8, 24).map { MachineMemory(ram * GIB, GpuMemory(it * GIB, unified = false)) } +
                    MachineMemory(ram * GIB, GpuMemory(ram * GIB * 3 / 4, unified = true))
            }
        val models =
            listOf(400 * MIB, 2 * GIB, 5 * GIB, 9 * GIB, 20 * GIB, 45 * GIB, 90 * GIB).flatMap { weights ->
                listOf(16L * 1024, 128 * 1024, 512 * 1024).flatMap { perToken ->
                    listOf(2048, 8192, 131_072, null).map { Model(weights, perToken, it) }
                }
            }
        var chosen = 0
        for (machine in machines) {
            for (model in models) {
                val config =
                    try {
                        choose(machine, model)
                    } catch (e: UserInputError) {
                        // Refused only when the shortest context on the CPU, which every machine has, is too much:
                        // more than its memory less the fifth, or the 2 GiB, that stay free.
                        val shortest = minOf(model.trained ?: MIN_AUTO_CONTEXT, MIN_AUTO_CONTEXT)
                        val system = checkNotNull(machine.systemBytes)
                        assertTrue(model.bytes(shortest) > system - maxOf(system / 5, 2 * GIB), "refused on $machine: ${e.message}")
                        continue
                    }
                chosen++
                val context = assertNotNull(config.contextLength)
                assertTrue(context <= (model.trained ?: MAX_AUTO_CONTEXT) && context <= MAX_AUTO_CONTEXT, "$config")
                val bytes = model.bytes(context)
                val share =
                    when (val gpu = config.gpu) {
                        GpuOffload.Off -> 0.0
                        GpuOffload.Max -> 1.0
                        is GpuOffload.Ratio -> gpu.value
                    }
                val gpu = machine.gpu
                val system = checkNotNull(machine.systemBytes)
                when {
                    gpu == null -> assertEquals(GpuOffload.Off, config.gpu)
                    gpu.unified -> assertTrue(bytes <= gpu.bytes || share == 0.0, "$config on $machine")
                    else -> assertTrue(bytes * share <= gpu.bytes, "$config on $machine")
                }
                val inSystem = if (gpu?.unified == false) bytes * (1 - share) else bytes.toDouble()
                assertTrue(inSystem <= system, "$config on $machine")
            }
        }
        // The table is not a list of refusals.
        assertTrue(chosen > machines.size * models.size / 2, "chosen: $chosen")
    }

    @Test
    fun aGpuCountsOnlyWhenTheEngineComputesOnIt() {
        val survey =
            HardwareSurvey(
                os = HostOs.LINUX,
                cpu = Cpu(null, "x86_64", null, null, 8),
                memoryBytes = 64 * GIB,
                gpus =
                    listOf(
                        Gpu("Intel GPU (8086:9a49)", null, unifiedMemory = false, GpuSource.SYSFS),
                        Gpu("NVIDIA GeForce RTX 3060", 12 * GIB, unifiedMemory = false, GpuSource.NVIDIA_SMI),
                        Gpu("NVIDIA GeForce RTX 4090", 24 * GIB, unifiedMemory = false, GpuSource.NVIDIA_SMI),
                    ),
            )

        fun engine(gpuOffload: Boolean) = EngineInfo("fake", "0", listOf("cpu"), gpuOffload)
        assertEquals(MachineMemory(64 * GIB, GpuMemory(24 * GIB, unified = false)), machineMemory(survey, engine(gpuOffload = true)))
        assertEquals(MachineMemory(64 * GIB), machineMemory(survey, engine(gpuOffload = false)))
        assertEquals(MachineMemory(64 * GIB), machineMemory(survey.copy(gpus = survey.gpus.take(1)), engine(gpuOffload = true)))
    }
}
