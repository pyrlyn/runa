package runa.daemon

import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.core.UserInputError
import runa.engine.EngineInfo
import runa.platform.HardwareSurvey
import kotlin.math.floor

// How a load with `auto` settles on a context length and a GPU share: by what the machine has,
// less a reserve for everything else that runs on it, and less what the loaded models hold.

/** The memory a load with `auto` plans with. */
public data class MachineMemory(
    /** The machine's memory; null when the OS does not tell. */
    val systemBytes: Long?,
    /** The GPU the engine computes on; null when there is none, or the engine cannot use it. */
    val gpu: GpuMemory? = null,
)

public data class GpuMemory(
    val bytes: Long,
    /** True when it is a share of the machine's memory, and not memory of the GPU's own. */
    val unified: Boolean,
)

/** What [survey] means to [engine]: a GPU counts only when the engine can put a model on one. */
public fun machineMemory(
    survey: HardwareSurvey,
    engine: EngineInfo,
): MachineMemory =
    MachineMemory(
        systemBytes = survey.memoryBytes,
        // llama.cpp can spread a model over several cards; planning with the largest alone is the cautious choice.
        gpu =
            survey.gpus
                .takeIf { engine.gpuOffload }
                ?.mapNotNull { gpu -> gpu.memoryBytes?.let { GpuMemory(it, gpu.unifiedMemory) } }
                ?.maxByOrNull { it.bytes },
    )

/** A loaded model, as far as the memory it holds goes. */
internal data class HeldModel(
    val bytes: Long,
    val gpu: GpuOffload,
)

/**
 * The configuration a load with `auto` gets. What the request fixed ([contextLength], [gpu]) stays,
 * and the rest is the first of these that fits:
 *
 * 1. Everything on the GPU, with the longest context that fits there: speed comes first.
 * 2. On a GPU with memory of its own, as much of the model as fits, with a context no longer than
 *    the default: memory is short then, and every token of context takes layers off the GPU.
 * 3. Everything on the CPU, with the longest context that fits.
 *
 * A context is never longer than the model was trained with ([trainedContext]) or than
 * [MAX_AUTO_CONTEXT], and none shorter than [MIN_AUTO_CONTEXT] is chosen.
 *
 * @param model its name, for the error message.
 * @param estimate the bytes the model takes with a context length.
 * @throws UserInputError when the model fits in no way, or the machine's memory is not known.
 */
internal fun chooseLoadConfig(
    model: String,
    memory: MachineMemory,
    held: List<HeldModel>,
    contextLength: Int?,
    gpu: GpuOffload?,
    trainedContext: Int?,
    estimate: (contextLength: Int) -> Long,
): LoadConfig {
    val budget = Budget(memory, held)
    val contexts = contextLength?.let(::listOf) ?: contexts(trainedContext)
    // An estimate reads the model file's header, and the same context is asked about more than once.
    val sizes = HashMap<Int, Long>()

    fun size(context: Int) = sizes.getOrPut(context) { estimate(context) }

    fun longest(share: Double) = contexts.firstOrNull { budget.fits(size(it), share) }

    if (gpu != null) {
        longest(gpu.share())?.let { return LoadConfig(it, gpu) }
    } else {
        if (budget.hasGpu) longest(1.0)?.let { return LoadConfig(it, GpuOffload.Max) }
        if (budget.gpuHasItsOwnMemory) {
            for (context in contexts.filter { it <= LoadConfig.DEFAULT_CONTEXT_LENGTH }.ifEmpty { contexts }) {
                val share = budget.largestShare(size(context))
                if (share >= MIN_SHARE && budget.fits(size(context), share)) return LoadConfig(context, GpuOffload.Ratio(share))
            }
        }
        longest(0.0)?.let { return LoadConfig(it, GpuOffload.Off) }
    }
    val shortest = contexts.last()
    throw UserInputError(
        "auto finds no way to fit $model: with a context of $shortest it takes about ${megabytes(size(shortest))} MB, " +
            "and this machine has ${budget.describe()} for it. Give gpu and context_length to load it anyway",
    )
}

/** The contexts to try, longest first. */
private fun contexts(trained: Int?): List<Int> {
    val longest = minOf(trained ?: MAX_AUTO_CONTEXT, MAX_AUTO_CONTEXT)
    return listOf(longest) + generateSequence(MAX_AUTO_CONTEXT) { it / 2 }.takeWhile { it >= MIN_AUTO_CONTEXT }.filter { it < longest }
}

private fun GpuOffload.share(): Double =
    when (this) {
        GpuOffload.Off -> 0.0
        GpuOffload.Max -> 1.0
        is GpuOffload.Ratio -> value
    }

/** What is free for one more model. */
private class Budget(
    memory: MachineMemory,
    held: List<HeldModel>,
) {
    private val system: Long
    private val onGpu: Long?

    /** True when the GPU has memory of its own, so a model can be split between it and the machine's. */
    val gpuHasItsOwnMemory: Boolean = memory.gpu?.unified == false
    val hasGpu: Boolean get() = onGpu != null

    init {
        val installed =
            memory.systemBytes
                ?: throw UserInputError(
                    "auto needs the size of this machine's memory, which the system does not tell; give gpu and context_length",
                )
        val heldOnGpu = if (gpuHasItsOwnMemory) held.sumOf { (it.bytes * it.gpu.share()).toLong() } else 0
        val heldInSystem = held.sumOf { it.bytes } - heldOnGpu
        system = installed - maxOf(installed / RESERVED_PART, MIN_RESERVE_BYTES) - heldInSystem
        onGpu =
            when {
                memory.gpu == null -> null

                // What the system lets the GPU have is its own advice already, and it is the machine's memory all the same.
                memory.gpu.unified -> minOf(memory.gpu.bytes - heldInSystem, system)

                else -> memory.gpu.bytes / PERCENT * GPU_USABLE_PERCENT - heldOnGpu
            }
    }

    /** Whether a model of [bytes] fits with [share] of it on the GPU. */
    fun fits(
        bytes: Long,
        share: Double,
    ): Boolean =
        when {
            onGpu == null || share == 0.0 -> bytes <= system
            !gpuHasItsOwnMemory -> bytes <= onGpu
            else -> bytes * share <= onGpu && bytes * (1 - share) <= system
        }

    /** The largest share of a model of [bytes] that the GPU's own memory takes, in hundredths. */
    fun largestShare(bytes: Long): Double = (floor((onGpu ?: 0).toDouble() / bytes * PERCENT) / PERCENT).coerceIn(0.0, 1.0)

    fun describe(): String =
        when {
            onGpu == null -> "${megabytes(system)} MB of memory"
            !gpuHasItsOwnMemory -> "${megabytes(onGpu)} MB of memory"
            else -> "${megabytes(system)} MB of memory and ${megabytes(onGpu)} MB on the GPU"
        }
}

private fun megabytes(bytes: Long): Long = bytes.coerceAtLeast(0) / BYTES_PER_MEGABYTE

/**
 * The longest context `auto` chooses. A longer one costs gigabytes of cache on a large model, so
 * it is asked for by its number.
 */
internal const val MAX_AUTO_CONTEXT: Int = 32_768

/** `auto` gives up on a placement before it goes below this: less is too little for a conversation. */
internal const val MIN_AUTO_CONTEXT: Int = 2_048

// A fifth of the machine's memory, and 2 GiB at least, stays with the system and the other programs.
private const val RESERVED_PART = 5
private const val MIN_RESERVE_BYTES = 2L * 1024 * 1024 * 1024

// A card also holds the desktop and the engine's compute buffers, which the estimate leaves out.
private const val GPU_USABLE_PERCENT = 90
private const val PERCENT = 100

// Less than this on the GPU does not pay for moving the data between it and the CPU.
private const val MIN_SHARE = 0.1

// Decimal, as the CLI shows sizes.
private const val BYTES_PER_MEGABYTE = 1_000_000L
