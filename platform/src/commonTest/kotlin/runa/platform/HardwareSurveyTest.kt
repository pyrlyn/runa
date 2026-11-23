package runa.platform

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** A machine that exists only as the answers of its sources. */
private class FakeMachine(
    override val os: HostOs,
    override val architecture: String = "x86_64",
    private val memory: Long? = 16L * GIB,
    private val processors: Int? = null,
    private val files: Map<String, String> = emptyMap(),
    private val sysctl: Map<String, Any> = emptyMap(),
    private val programs: Map<String, String> = emptyMap(),
    private val metal: List<MetalDevice> = emptyList(),
) : SystemSources {
    /** Every program the survey started. */
    val started = mutableListOf<String>()

    override fun memoryBytes(): Long? = memory

    override fun processorCount(): Int? = processors

    override fun file(path: String): String? = files[path]

    override fun directory(path: String): List<String> =
        files.keys
            .filter { it.startsWith("$path/") }
            .map { it.removePrefix("$path/").substringBefore('/') }
            .distinct()

    override fun sysctlText(name: String): String? = sysctl[name] as? String

    override fun sysctlNumber(name: String): Long? = (sysctl[name] as? Number)?.toLong()

    override fun output(command: String): String? {
        started += command
        return programs[command.substringBefore(' ')]
    }

    override fun metalDevices(): List<MetalDevice> = metal
}

private const val GIB = 1024L * 1024L * 1024L
private const val CPU = "/sys/devices/system/cpu"
private const val DRM = "/sys/class/drm"

class HardwareSurveyTest {
    @Test
    fun anAppleSiliconMac() {
        val mac =
            FakeMachine(
                os = HostOs.MACOS,
                architecture = "arm64",
                memory = 36L * GIB,
                sysctl =
                    mapOf(
                        "machdep.cpu.brand_string" to "Apple M3 Pro",
                        "hw.physicalcpu" to 12,
                        "hw.perflevel0.physicalcpu" to 6,
                        "hw.logicalcpu" to 12,
                    ),
                metal = listOf(MetalDevice("Apple M3 Pro", 27L * GIB, hasUnifiedMemory = true)),
            )
        assertEquals(
            HardwareSurvey(
                os = HostOs.MACOS,
                cpu = Cpu("Apple M3 Pro", "arm64", cores = 12, performanceCores = 6, threads = 12),
                memoryBytes = 36L * GIB,
                gpus = listOf(Gpu("Apple M3 Pro", 27L * GIB, unifiedMemory = true, GpuSource.METAL)),
            ),
            surveyHardware(mac),
        )
        assertEquals(emptyList(), mac.started)
    }

    @Test
    fun anIntelMacHasNoPerformanceCoresAndAGpuOfItsOwn() {
        val mac =
            FakeMachine(
                os = HostOs.MACOS,
                sysctl =
                    mapOf(
                        "machdep.cpu.brand_string" to "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz",
                        "hw.physicalcpu" to 8,
                        "hw.logicalcpu" to 16,
                    ),
                metal = listOf(MetalDevice("AMD Radeon Pro 5500M", 8L * GIB, hasUnifiedMemory = false)),
            )
        val survey = surveyHardware(mac)
        assertEquals(
            Cpu("Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz", "x86_64", cores = 8, performanceCores = null, threads = 16),
            survey.cpu,
        )
        assertEquals(listOf(Gpu("AMD Radeon Pro 5500M", 8L * GIB, unifiedMemory = false, GpuSource.METAL)), survey.gpus)
    }

    @Test
    fun aLinuxMachineWithAnNvidiaCard() {
        val linux =
            FakeMachine(
                os = HostOs.LINUX,
                memory = 64L * GIB,
                processors = 4,
                files =
                    mapOf(
                        "/proc/cpuinfo" to "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\n",
                        // Two cores of two hyper-threads each.
                        "$CPU/cpu0/topology/thread_siblings_list" to "0,2\n",
                        "$CPU/cpu1/topology/thread_siblings_list" to "1,3\n",
                        "$CPU/cpu2/topology/thread_siblings_list" to "0,2\n",
                        "$CPU/cpu3/topology/thread_siblings_list" to "1,3\n",
                        "$CPU/cpufreq/boost" to "1\n",
                        "$DRM/card0/device/vendor" to "0x10de\n",
                        "$DRM/card0/device/device" to "0x2684\n",
                        "$DRM/card0-HDMI-A-1/status" to "connected\n",
                        "$DRM/renderD128/device/vendor" to "0x10de\n",
                    ),
                programs = mapOf("nvidia-smi" to "NVIDIA GeForce RTX 4090, 24564\n"),
            )
        assertEquals(
            HardwareSurvey(
                os = HostOs.LINUX,
                cpu = Cpu("AMD Ryzen 9 7950X 16-Core Processor", "x86_64", cores = 2, performanceCores = null, threads = 4),
                memoryBytes = 64L * GIB,
                // Once, from NVIDIA's tool: the kernel's entry for the same card is left out.
                gpus = listOf(Gpu("NVIDIA GeForce RTX 4090", 24_564L * 1024 * 1024, unifiedMemory = false, GpuSource.NVIDIA_SMI)),
            ),
            surveyHardware(linux),
        )
        assertEquals(listOf("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"), linux.started)
    }

    @Test
    fun aLinuxMachineTellsItsOtherCardsThroughSysfs() {
        val linux =
            FakeMachine(
                os = HostOs.LINUX,
                files =
                    mapOf(
                        "$DRM/card0/device/vendor" to "0x8086\n",
                        "$DRM/card0/device/device" to "0x9a49\n",
                        "$DRM/card1/device/vendor" to "0x1002\n",
                        "$DRM/card1/device/device" to "0x73bf\n",
                        "$DRM/card1/device/mem_info_vram_total" to "17163091968\n",
                        // The driver is there and NVIDIA's tool is not.
                        "$DRM/card2/device/vendor" to "0x10de\n",
                        // A virtual machine's display.
                        "$DRM/card3/device/vendor" to "0x1af4\n",
                    ),
            )
        assertEquals(
            listOf(
                Gpu("Intel GPU (8086:9a49)", null, unifiedMemory = false, GpuSource.SYSFS),
                Gpu("AMD GPU (1002:73bf)", 17_163_091_968, unifiedMemory = false, GpuSource.SYSFS),
                Gpu("NVIDIA GPU", null, unifiedMemory = false, GpuSource.SYSFS),
            ),
            surveyHardware(linux).gpus,
        )
    }

    @Test
    fun anArmLinuxMachineWithoutANameOrAGpu() {
        val linux =
            FakeMachine(
                os = HostOs.LINUX,
                architecture = "arm64",
                memory = null,
                processors = 4,
                files = mapOf("/proc/cpuinfo" to "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd0c\n"),
                // A card that NVIDIA's tool cannot tell the memory of.
                programs = mapOf("nvidia-smi" to "Tesla T4, [N/A]\n"),
            )
        assertEquals(
            HardwareSurvey(
                os = HostOs.LINUX,
                cpu = Cpu(model = null, "arm64", cores = null, performanceCores = null, threads = 4),
                memoryBytes = null,
                gpus = listOf(Gpu("Tesla T4", null, unifiedMemory = false, GpuSource.NVIDIA_SMI)),
            ),
            surveyHardware(linux),
        )
    }

    @Test
    fun windowsTellsItsMemoryAndItsProcessors() {
        val survey = surveyHardware(FakeMachine(os = HostOs.WINDOWS, processors = 16))
        assertEquals(HardwareSurvey(HostOs.WINDOWS, Cpu(null, "x86_64", null, null, threads = 16), 16L * GIB, emptyList()), survey)
    }

    @Test
    fun thisMachineAnswers() {
        val survey = surveyHardware()
        assertEquals(hostOs, survey.os)
        assertTrue(survey.cpu.architecture in setOf("arm64", "x86_64"), survey.cpu.architecture)
        assertTrue(assertNotNull(survey.cpu.threads) >= 1)
        assertTrue(survey.gpus.all { it.name.isNotBlank() }, "${survey.gpus}")
        if (survey.os == HostOs.MACOS) assertNotNull(survey.cpu.model)
    }

    @Test
    fun reportsAPlausibleCoreCount() {
        val cores = computeCoreCount()
        assertTrue(cores in 1..4096, "cores: $cores")
        // Never more than the OS schedules on.
        assertTrue(cores <= (systemSources().processorCount() ?: cores), "cores: $cores")
    }

    @Test
    fun aProgramThatIsMissingOrFailsHasNoOutput() {
        val sources = systemSources()
        if (sources.os == HostOs.WINDOWS) return
        assertEquals("one\ntwo\n", sources.output("printf 'one\\ntwo\\n'"))
        assertEquals(null, sources.output("runa-no-such-program --version"))
        assertEquals(null, sources.output("false"))
    }
}
