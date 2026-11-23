package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.mordant.widgets.Text
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.requireRunaBackend
import runa.cli.output.formatBytes
import runa.core.api.EngineDescription
import runa.engine.EngineInfo
import runa.platform.Cpu
import runa.platform.Gpu
import runa.platform.HardwareSurvey

/** `runa runtime survey --json`. A null field is something the OS did not tell. */
@Serializable
internal data class SurveyView(
    val os: String,
    val cpu: CpuView,
    @SerialName("memory_bytes") val memoryBytes: Long?,
    /** Every GPU that was found, whether or not [engine] can compute on it. */
    val gpus: List<GpuView>,
    /** The engine in this binary; `gpu_offload` tells whether it can put a model on one of [gpus]. */
    val engine: EngineDescription,
) {
    constructor(hardware: HardwareSurvey, engine: EngineInfo) : this(
        os = hardware.os.name.lowercase(),
        cpu = CpuView(hardware.cpu),
        memoryBytes = hardware.memoryBytes,
        gpus = hardware.gpus.map(::GpuView),
        engine = EngineDescription(engine.name, engine.version, engine.accelerators, engine.gpuOffload),
    )

    fun describe(): String =
        listOfNotNull(
            "CPU: ${cpu.describe()}",
            "Memory: ${memoryBytes?.let(::formatBytes) ?: "unknown"}",
            if (gpus.isEmpty()) "GPU: none found" else null,
        ).plus(gpus.map { "GPU: ${it.describe()}" })
            .plus("Engine: ${engine.describe()}")
            .plus(
                listOfNotNull(
                    "This build of runa computes on the CPU only".takeIf { gpus.isNotEmpty() && !engine.gpuOffload },
                ),
            ).joinToString("\n")
}

@Serializable
internal data class CpuView(
    val model: String?,
    val architecture: String,
    /** Physical cores. */
    val cores: Int?,
    /** The fast cores of a CPU that also has slow ones. */
    @SerialName("performance_cores") val performanceCores: Int?,
    /** The processors the OS schedules on, hyper-threads included. */
    val threads: Int?,
) {
    constructor(cpu: Cpu) : this(cpu.model, cpu.architecture, cpu.cores, cpu.performanceCores, cpu.threads)

    /** `Apple M3 Pro (arm64), 12 cores (6 performance), 12 threads`. */
    fun describe(): String =
        listOfNotNull(
            model?.let { "$it ($architecture)" } ?: architecture,
            cores?.let { "$it cores" + performanceCores?.let { fast -> " ($fast performance)" }.orEmpty() },
            threads?.let { "$it threads" },
        ).joinToString(", ")
}

@Serializable
internal data class GpuView(
    val name: String,
    /** Its own memory, or with [unifiedMemory] the share of the machine's memory the system lets it have. */
    @SerialName("memory_bytes") val memoryBytes: Long?,
    @SerialName("unified_memory") val unifiedMemory: Boolean,
    /** Who told about it: `metal`, `nvidia-smi` or `sysfs`. */
    val source: String,
) {
    constructor(gpu: Gpu) : this(gpu.name, gpu.memoryBytes, gpu.unifiedMemory, gpu.source.id)

    /** `Apple M3 Pro, 27.0 GB of the machine's memory (metal)`. */
    fun describe(): String =
        listOfNotNull(name, memoryBytes?.let { formatBytes(it) + if (unifiedMemory) " of the machine's memory" else "" })
            .joinToString(", ") + " ($source)"
}

/** `llama.cpp b6100 (cpu, metal)`. */
internal fun EngineDescription.describe(): String = "$name $version (${accelerators.joinToString(", ")})"

internal class RuntimeCommand : CliktCommand(name = "runtime") {
    init {
        subcommands(Survey())
    }

    override val helpTags: Map<String, String> = CommandGroup.RUNTIME.helpTags
    override val printHelpOnEmptyArgs: Boolean = true

    override fun help(context: Context): String = "Show what this machine and this build run models with"

    override fun run() = Unit

    private class Survey : CliktCommand(name = "survey") {
        private val globals by requireObject<Globals>()
        private val json by option("--json", help = JSON_HELP).flag()

        override fun help(context: Context): String =
            """
            Show the CPU, the memory, the GPUs and what the engine computes on

            It is the machine this command runs on, and it asks no daemon. A GPU counts for `runa load`
            only when the engine was built to compute on it: the survey says so when it was not.
            `runa load --auto` chooses by this survey, taken by the daemon on its own machine.
            """.trimIndent()

        override fun run() {
            requireRunaBackend(globals)
            val view = SurveyView(globals.environment.hardware(), globals.engine.info)
            globals.renderer(json).render(view, SurveyView.serializer()) { Text(it.describe()) }
        }
    }
}
