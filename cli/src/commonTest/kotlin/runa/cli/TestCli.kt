package runa.cli

import com.github.ajalt.clikt.core.CoreCliktCommand
import com.github.ajalt.mordant.rendering.AnsiLevel
import com.github.ajalt.mordant.terminal.Terminal
import com.github.ajalt.mordant.terminal.TerminalRecorder
import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import runa.platform.HostOs
import runa.platform.systemEnv
import kotlin.random.Random
import kotlin.test.assertEquals

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
    ): CliRun {
        val recorder = TerminalRecorder(ansiLevel = AnsiLevel.NONE, width = 80)
        val environment =
            CliEnvironment(
                version = "1.2.3",
                gitSha = "abc1234",
                env = (env + ("RUNA_HOME" to home.toString()))::get,
                fileSystem = FileSystem.SYSTEM,
                os = HostOs.LINUX,
                terminal = Terminal(terminalInterface = recorder),
            )
        val exitCode = runCli(args.toList(), environment, extraCommands)
        return CliRun(exitCode, recorder.stdout(), recorder.stderr())
    }

    companion object {
        fun temporaryHome(): Path = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-cli-test-${Random.nextLong().toULong()}"
    }
}

/**
 * Compares [actual] with `src/commonTest/fixtures/<name>` (Gradle runs native tests from the
 * module directory). `RUNA_UPDATE_GOLDEN=1 ./gradlew :cli:allTests` rewrites the files instead.
 */
internal fun assertGolden(
    name: String,
    actual: String,
) {
    val file = "src/commonTest/fixtures".toPath() / name
    if (systemEnv("RUNA_UPDATE_GOLDEN") == "1") {
        FileSystem.SYSTEM.write(file) { writeUtf8(actual) }
        return
    }
    val expected = FileSystem.SYSTEM.read(file) { readUtf8() }
    assertEquals(expected, actual, "Output differs from $file; rerun with RUNA_UPDATE_GOLDEN=1 if the change is intended")
}
