package runa.platform

/** What this machine has to run a model with. A null field is something the OS did not tell. */
public data class HardwareSurvey(
    val os: HostOs,
    val cpu: Cpu,
    val memoryBytes: Long?,
    /** Every GPU that was found, whether or not this binary can compute on it. */
    val gpus: List<Gpu>,
)

public data class Cpu(
    /** The name its maker gave it, for example `Apple M3 Pro`. */
    val model: String?,
    /** `arm64` or `x86_64`. */
    val architecture: String,
    /** Physical cores. */
    val cores: Int?,
    /** The fast cores of a CPU that also has slow ones; null when all cores are alike. */
    val performanceCores: Int?,
    /** The processors the OS schedules on, hyper-threads included. */
    val threads: Int?,
)

public data class Gpu(
    val name: String,
    /**
     * The memory a model can have on it: its own memory, or with [unifiedMemory] the share of the
     * machine's memory that the system lets a process give the GPU.
     */
    val memoryBytes: Long?,
    /** True when the GPU has no memory of its own and works in the machine's. */
    val unifiedMemory: Boolean,
    val source: GpuSource,
)

/** Who told about a GPU. */
public enum class GpuSource(
    public val id: String,
) {
    METAL("metal"),
    NVIDIA_SMI("nvidia-smi"),
    SYSFS("sysfs"),
}

/** Asks [sources] what the machine has. It starts `nvidia-smi` on Linux, so it is not for a hot path. */
public fun surveyHardware(sources: SystemSources = systemSources()): HardwareSurvey =
    HardwareSurvey(
        os = sources.os,
        cpu = surveyCpu(sources),
        memoryBytes = sources.memoryBytes(),
        gpus =
            when (sources.os) {
                HostOs.MACOS -> {
                    sources.metalDevices().map {
                        Gpu(it.name, it.recommendedMaxWorkingSetSize.positive(), it.hasUnifiedMemory, GpuSource.METAL)
                    }
                }

                HostOs.LINUX -> {
                    linuxGpus(sources)
                }

                HostOs.WINDOWS -> {
                    emptyList()
                }
            },
    )

/** The processor alone: cheap, and all that choosing a thread count needs. */
public fun surveyCpu(sources: SystemSources): Cpu =
    when (sources.os) {
        HostOs.MACOS -> {
            Cpu(
                model = sources.sysctlText("machdep.cpu.brand_string"),
                architecture = sources.architecture,
                cores = sources.sysctlCount("hw.physicalcpu"),
                // perflevel0 is the performance cluster; Intel Macs do not have the key.
                performanceCores = sources.sysctlCount("hw.perflevel0.physicalcpu"),
                threads = sources.sysctlCount("hw.logicalcpu") ?: sources.processorCount(),
            )
        }

        HostOs.LINUX -> {
            Cpu(
                model = linuxCpuModel(sources),
                architecture = sources.architecture,
                cores = linuxCores(sources),
                performanceCores = null,
                threads = sources.processorCount(),
            )
        }

        // Telling cores from hyper-threads needs GetLogicalProcessorInformationEx, which is not
        // worth its code until Windows is a supported target (PLAN.md, section 13).
        HostOs.WINDOWS -> {
            Cpu(
                model = null,
                architecture = sources.architecture,
                cores = null,
                performanceCores = null,
                threads = sources.processorCount(),
            )
        }
    }

private fun SystemSources.sysctlCount(name: String): Int? = sysctlNumber(name)?.takeIf { it in 1..Int.MAX_VALUE }?.toInt()

private fun Long.positive(): Long? = takeIf { it > 0 }

/** x86 kernels name the processor in `/proc/cpuinfo`; most ARM kernels give only part numbers there. */
private fun linuxCpuModel(sources: SystemSources): String? =
    sources
        .file("/proc/cpuinfo")
        ?.lineSequence()
        ?.firstOrNull { it.startsWith("model name") }
        ?.substringAfter(':')
        ?.trim()
        ?.takeIf { it.isNotEmpty() }

/** Hyper-threads of one core list the same siblings, so the distinct lists are the cores. */
private fun linuxCores(sources: SystemSources): Int? =
    sources
        .directory(CPU_DIR)
        .filter { it.startsWith("cpu") && it.drop(3).toIntOrNull() != null }
        .mapNotNull { sources.file("$CPU_DIR/$it/topology/thread_siblings_list")?.trim() }
        .distinct()
        .size
        .takeIf { it > 0 }

/** NVIDIA's own tool knows its cards best; the kernel's list of display adapters has the others. */
private fun linuxGpus(sources: SystemSources): List<Gpu> {
    val nvidia = nvidiaGpus(sources)
    return nvidia + sysfsGpus(sources, skipNvidia = nvidia.isNotEmpty())
}

/** A line per card: `NVIDIA GeForce RTX 4090, 24564`, the memory in MiB. */
private fun nvidiaGpus(sources: SystemSources): List<Gpu> =
    sources
        .output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
        .orEmpty()
        .lines()
        .filter { ',' in it }
        .map { line ->
            Gpu(
                name = line.substringBeforeLast(',').trim(),
                memoryBytes =
                    line
                        .substringAfterLast(',')
                        .trim()
                        .toLongOrNull()
                        ?.let { it * BYTES_PER_MIB }
                        ?.positive(),
                unifiedMemory = false,
                source = GpuSource.NVIDIA_SMI,
            )
        }

private fun sysfsGpus(
    sources: SystemSources,
    skipNvidia: Boolean,
): List<Gpu> =
    sources
        .directory(DRM_DIR)
        // `card0-HDMI-A-1` is an output of a card and `renderD128` another name of it.
        .filter { CARD.matches(it) }
        .sorted()
        .mapNotNull { card ->
            val device = "$DRM_DIR/$card/device"
            val vendor = sources.file("$device/vendor")?.pciId()
            // An adapter of any other maker is a virtual or display-only one, which runs no model.
            val maker = PCI_VENDORS[vendor] ?: return@mapNotNull null
            if (vendor == NVIDIA && skipNvidia) return@mapNotNull null
            // sysfs has no names, only numbers: the maker and the PCI id are what can be said.
            val id =
                sources
                    .file("$device/device")
                    ?.pciId()
                    ?.let { " ($vendor:$it)" }
                    .orEmpty()
            Gpu(
                name = "$maker GPU$id",
                // Only the amdgpu driver has this file.
                memoryBytes =
                    sources
                        .file("$device/mem_info_vram_total")
                        ?.trim()
                        ?.toLongOrNull()
                        ?.positive(),
                unifiedMemory = false,
                source = GpuSource.SYSFS,
            )
        }

/** `0x10de` and a line end become `10de`. */
private fun String.pciId(): String = trim().removePrefix("0x")

private const val CPU_DIR = "/sys/devices/system/cpu"
private const val DRM_DIR = "/sys/class/drm"
private val CARD = Regex("card[0-9]+")
private const val NVIDIA = "10de"
private val PCI_VENDORS = mapOf(NVIDIA to "NVIDIA", "1002" to "AMD", "8086" to "Intel")
private const val BYTES_PER_MIB = 1024L * 1024L
