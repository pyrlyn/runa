package runa.cli

import runa.engine.EngineInfo
import runa.engine.testing.FakeEngine
import runa.platform.Cpu
import runa.platform.Gpu
import runa.platform.GpuSource
import runa.platform.HardwareSurvey
import runa.platform.HostOs
import runa.testing.assertGolden
import kotlin.test.Test
import kotlin.test.assertEquals

class RuntimeCommandTest {
    private val cli = TestCli()

    @Test
    fun surveyShowsTheMachineAndTheEngine() {
        val run = cli.run("runtime", "survey")

        assertEquals(0, run.exitCode)
        assertGolden("runtime-survey.txt", run.stdout)
        assertGolden("runtime-survey.json", cli.run("runtime", "survey", "--json").stdout)
        assertEquals("", run.stderr)
    }

    @Test
    fun surveySaysWhatTheSystemDidNotTell() {
        val bare =
            HardwareSurvey(
                HostOs.LINUX,
                Cpu(model = null, "arm64", cores = null, performanceCores = null, threads = 4),
                memoryBytes = null,
                gpus = emptyList(),
            )

        assertEquals(
            """
            CPU: arm64, 4 threads
            Memory: unknown
            GPU: none found
            Engine: fake 0 (cpu)

            """.trimIndent(),
            cli.run("runtime", "survey", hardware = bare).stdout,
        )
        assertEquals(
            """{"os":"linux","cpu":{"model":null,"architecture":"arm64","cores":null,"performance_cores":null,"threads":4},""" +
                """"memory_bytes":null,"gpus":[],"engine":{"name":"fake","version":"0","accelerators":["cpu"],"gpu_offload":false}}""" +
                "\n",
            cli.run("runtime", "survey", "--json", hardware = bare).stdout,
        )
    }

    @Test
    fun surveySaysWhenTheGpusAreNotUsed() {
        val linux =
            HardwareSurvey(
                os = HostOs.LINUX,
                cpu = Cpu("AMD Ryzen 9 7950X 16-Core Processor", "x86_64", cores = 16, performanceCores = null, threads = 32),
                memoryBytes = 64_000_000_000,
                gpus =
                    listOf(
                        Gpu("NVIDIA GeForce RTX 4090", 24_000_000_000, unifiedMemory = false, GpuSource.NVIDIA_SMI),
                        Gpu("Intel GPU (8086:9a49)", null, unifiedMemory = false, GpuSource.SYSFS),
                    ),
            )

        assertEquals(
            """
            CPU: AMD Ryzen 9 7950X 16-Core Processor (x86_64), 16 cores, 32 threads
            Memory: 64.0 GB
            GPU: NVIDIA GeForce RTX 4090, 24.0 GB (nvidia-smi)
            GPU: Intel GPU (8086:9a49) (sysfs)
            Engine: fake 0 (cpu)
            This build of runa computes on the CPU only

            """.trimIndent(),
            cli.run("runtime", "survey", hardware = linux).stdout,
        )
    }

    @Test
    fun runtimeAloneAndHelpExplainTheCommand() {
        assertEquals(cli.run("runtime", "--help").stdout, cli.run("runtime").stdout)
        assertGolden("runtime-help.txt", cli.run("runtime", "--help").stdout)
        assertGolden("runtime-survey-help.txt", cli.run("runtime", "survey", "--help").stdout)
    }
}
