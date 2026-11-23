package runa.cli

import com.github.ajalt.clikt.core.CoreCliktCommand
import com.github.ajalt.mordant.rendering.AnsiLevel
import com.github.ajalt.mordant.terminal.Terminal
import com.github.ajalt.mordant.terminal.TerminalRecorder
import io.ktor.client.engine.HttpClientEngine
import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import runa.config.DaemonSettings
import runa.engine.Engine
import runa.engine.testing.FakeEngine
import runa.gguf.GgufWriter
import runa.platform.Cpu
import runa.platform.Gpu
import runa.platform.GpuSource
import runa.platform.HardwareSurvey
import runa.platform.HostOs
import kotlin.random.Random
import kotlin.time.Duration.Companion.milliseconds

internal class CliRun(
    val exitCode: Int,
    val stdout: String,
    val stderr: String,
)

/** Runs the whole CLI in-process against a recorded 80-column terminal without colours. */
internal class TestCli(
    /** `RUNA_HOME`. The default never exists, so tests that only read see pure defaults. */
    val home: Path = "/runa-home".toPath(),
) {
    fun run(
        vararg args: String,
        env: Map<String, String> = emptyMap(),
        extraCommands: List<CoreCliktCommand> = emptyList(),
        /** Lines typed at prompts; giving any makes the terminal interactive. */
        input: List<String> = emptyList(),
        /** False for [input] that is piped in, with nobody at the keyboard. */
        interactive: Boolean = input.isNotEmpty(),
        /** Answers the CLI's HTTP requests; without one, a request reaches the real network. */
        http: HttpClientEngine? = null,
        memory: Long? = 16_000_000_000,
        hardware: HardwareSurvey = TEST_HARDWARE,
        engine: Engine = FakeEngine(),
        /** Stands in for the daemon process: its HTTP API, its start and its lock. */
        daemon: FakeDaemon? = null,
        /** What `runa __daemon` runs. */
        daemonProcess: (DaemonSettings) -> Int = { error("this test does not run the daemon process") },
    ): CliRun {
        val recorder = TerminalRecorder(ansiLevel = AnsiLevel.NONE, width = 80, inputInteractive = interactive)
        recorder.inputLines = input.toMutableList()
        val environment =
            CliEnvironment(
                version = "1.2.3",
                gitSha = "abc1234",
                env = (env + ("RUNA_HOME" to home.toString()))::get,
                fileSystem = FileSystem.SYSTEM,
                os = HostOs.LINUX,
                terminal = Terminal(terminalInterface = recorder),
                httpEngine = daemon?.http ?: http,
                physicalMemory = { memory },
                hardware = { hardware },
                engine = { engine },
                daemon = daemonProcess,
                startDaemon = { arguments, _ -> checkNotNull(daemon) { "this test has no daemon to start" }.start(arguments) },
                daemonLockHeld = { daemon?.running == true },
                // Long enough for a fake that starts at once, short enough for one that never does.
                daemonStartTimeout = 300.milliseconds,
                downloadPoll = 1.milliseconds,
            )
        val exitCode = runCli(args.toList(), environment, extraCommands)
        return CliRun(exitCode, recorder.stdout(), recorder.stderr())
    }

    companion object {
        // Canonical, because import resolves links and macOS keeps its temporary directory behind one.
        fun temporaryHome(): Path =
            FileSystem.SYSTEM.canonicalize(FileSystem.SYSTEM_TEMPORARY_DIRECTORY) / "runa-cli-test-${Random.nextLong().toULong()}"
    }
}

/** The machine the tests run on, as far as `runtime survey` can tell. */
internal val TEST_HARDWARE =
    HardwareSurvey(
        os = HostOs.MACOS,
        cpu = Cpu("Apple M3 Pro", "arm64", cores = 12, performanceCores = 6, threads = 12),
        memoryBytes = 36_000_000_000,
        gpus = listOf(Gpu("Apple M3 Pro", 27_000_000_000, unifiedMemory = true, GpuSource.METAL)),
    )

/** A header-only language model: all that `ls` and `import` ever read of a model file. */
internal fun tinyLlama(fileType: Long = 15): GgufWriter =
    GgufWriter()
        .put("general.architecture", "llama")
        .put("general.name", "Tiny Llama")
        .put("general.file_type", fileType)
        .put("llama.context_length", 2048)
        .tensor("token_embd.weight", 1_000, 1_240)

internal fun writeModel(
    path: Path,
    model: GgufWriter = tinyLlama(),
): Path {
    FileSystem.SYSTEM.createDirectories(checkNotNull(path.parent))
    FileSystem.SYSTEM.write(path) { model.writeTo(this) }
    return path
}
