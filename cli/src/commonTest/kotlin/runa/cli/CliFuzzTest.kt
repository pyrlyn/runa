package runa.cli

import com.github.ajalt.clikt.core.BaseCliktCommand
import com.github.ajalt.clikt.core.CliktCommand
import okio.FileSystem
import runa.core.ExitCode
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Property test of the command line (PLAN.md, phase 11). Whatever is typed, runa answers with
 * success, with a usage mistake or with a backend error; an unexpected error, which prints a
 * stack trace, is a bug. Every case comes from its seed, so a failure names the case to run again.
 */
class CliFuzzTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val modelFile = cli.home / "models" / "acme" / "tiny" / "tiny-Q4_K_M.gguf"

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    /** One command of the tree: the words that lead to it and its options, with whether each takes a value. */
    private class Node(
        val path: List<String>,
        val options: List<Pair<String, Boolean>>,
    )

    /** Reads the command tree from inside it, so a new command or option is fuzzed without a change here. */
    private class Probe : CliktCommand(name = PROBE) {
        val nodes = mutableListOf<Node>()

        override fun run() = collect(currentContext.findRoot().command, emptyList())

        private fun collect(
            command: BaseCliktCommand<*>,
            path: List<String>,
        ) {
            // Help and the version end a command before it does anything; ODD has them.
            val options =
                command
                    .registeredOptions()
                    .flatMap { option -> option.names.map { it to (option.nvalues.last > 0) } }
                    .filter { (name, _) -> name !in ENDING }
            nodes += Node(path, options)
            for (child in command.registeredSubcommands()) if (child.commandName != PROBE) collect(child, path + child.commandName)
        }
    }

    private fun nodes(): List<Node> {
        val probe = Probe()
        cli.run(PROBE, extraCommands = listOf(probe))
        return probe.nodes.toList()
    }

    private fun seedCommands(): List<List<String>> =
        SEED_COMMANDS.map { line ->
            line.split(' ').map {
                if (it ==
                    MODEL_FILE
                ) {
                    modelFile.toString()
                } else {
                    it
                }
            }
        }

    // A seed command that the CLI no longer understands would only test the usage message.
    @Test
    fun theSeedCommandsAreUnderstood() {
        val refused =
            seedCommands().mapNotNull { arguments ->
                FileSystem.SYSTEM.deleteRecursively(cli.home)
                writeModel(modelFile)
                val run =
                    cli.run(
                        *arguments.toTypedArray(),
                        daemon = FakeDaemon(cli.home).apply { start() },
                        daemonProcess = { ExitCode.OK },
                    )
                "runa ${arguments.joinToString(" ")}: ${run.exitCode}: ${run.stderr}".takeIf {
                    run.exitCode !in
                        setOf(ExitCode.OK, ExitCode.BACKEND)
                }
            }
        assertEquals(emptyList(), refused)
    }

    @Test
    fun whateverIsTypedIsAnsweredWithoutAnUnexpectedError() {
        val nodes = nodes()
        assertTrue(nodes.size > 20, "the command tree was not read: ${nodes.map { it.path }}")
        val values = VALUES + listOf(cli.home.toString(), modelFile.toString(), (cli.home / "nothing.gguf").toString())
        val seeds = seedCommands()
        val outcomes = mutableMapOf<Int, Int>()

        repeat(CASES) { seed ->
            val random = Random(seed)
            val arguments = random.arguments(nodes, seeds, values)
            val environment = if (random.nextInt(8) == 0) mapOf(VARIABLES.random(random) to values.random(random)) else emptyMap()

            // Every case starts from the same directory, so its seed alone brings it back.
            FileSystem.SYSTEM.deleteRecursively(cli.home)
            if (random.nextInt(4) != 0) writeModel(modelFile)
            val daemon = FakeDaemon(cli.home)
            if (random.nextInt(4) != 0) daemon.start()

            val case = "seed $seed: runa ${arguments.joinToString(" ") { "'$it'" }} with $environment"
            val run =
                try {
                    cli.run(*arguments.toTypedArray(), env = environment, daemon = daemon, daemonProcess = { ExitCode.OK })
                } catch (e: Exception) {
                    fail("$case threw ${e.stackTraceToString().take(TOLD)}")
                }
            val told = "$case exited with ${run.exitCode}: ${run.stderr.take(TOLD)}"
            assertTrue(run.exitCode in setOf(ExitCode.OK, ExitCode.USER_INPUT, ExitCode.BACKEND), told)
            assertTrue("    at " !in run.stderr, told)
            outcomes[run.exitCode] = (outcomes[run.exitCode] ?: 0) + 1
        }
        // Cases that all end the same way would say that they stop at the first word.
        for (outcome in listOf(ExitCode.OK, ExitCode.USER_INPUT, ExitCode.BACKEND)) {
            assertTrue((outcomes[outcome] ?: 0) >= CASES / 100, "too few cases exit with $outcome: $outcomes")
        }
    }

    private fun Random.arguments(
        nodes: List<Node>,
        seeds: List<List<String>>,
        values: List<String>,
    ): List<String> {
        fun word() =
            when (nextInt(4)) {
                0 -> nodes.random(this).path.lastOrNull() ?: "--help"
                1 -> ODD.random(this)
                else -> values.random(this)
            }

        fun option(node: Node): List<String> {
            val (name, takesValue) = node.options.randomOrNull(this) ?: return emptyList()
            return when {
                !takesValue -> listOf(name)
                nextInt(4) == 0 -> listOf("$name=${values.random(this)}")
                else -> listOf(name, values.random(this))
            }
        }

        val arguments =
            when (nextInt(8)) {
                0 -> {
                    List(nextInt(1, 7)) { word() }
                }

                // A command of the tree with its own options: every command and option is reached,
                // the new ones too.
                in 1..3 -> {
                    val node = nodes.random(this)
                    val own = List(nextInt(3)) { option(node) }.flatten() + List(nextInt(3)) { word() }
                    List(nextInt(2)) { option(nodes.first()) }.flatten() + node.path + own.shuffled(this)
                }

                // A command that works, damaged a little or not at all: it gets past the usage
                // check, to what the command does with its values.
                else -> {
                    seeds.random(this)
                }
            }.toMutableList()
        repeat(nextInt(3)) {
            val at = if (arguments.isEmpty()) 0 else nextInt(arguments.size)
            when (nextInt(6)) {
                0 -> if (arguments.isNotEmpty()) arguments.removeAt(at)
                1 -> if (arguments.isNotEmpty()) arguments.add(at, arguments[at])
                2 -> if (arguments.isNotEmpty()) arguments[at] = word()
                3 -> arguments.addAll(at, option(nodes.random(this)))
                4 -> arguments.add(at, word())
                else -> arguments.shuffle(this)
            }
        }
        return arguments
    }

    private companion object {
        const val CASES = 3_000
        const val PROBE = "fuzz-probe"
        const val MODEL_FILE = "MODEL_FILE"
        val ENDING = setOf("-h", "--help", "-v", "--version")

        // One line for each thing runa does, as somebody would type it.
        val SEED_COMMANDS =
            listOf(
                "ls",
                "ls --json --variants",
                "ls acme/tiny",
                "ps --json",
                "status",
                "version --json",
                "config get",
                "config get port --json",
                "config set port 4000",
                "config path",
                "load alpha",
                "load alpha --gpu max --context-length 4096 --ttl 300 --parallel 2 --identifier a",
                "load beta --auto --estimate-only --json",
                "unload --all",
                "run tiny Hello",
                "run acme/tiny Hello --temp 0 --seed 1 --max-tokens 8 --stats",
                "run $MODEL_FILE Hello --json",
                "chat alpha -p Hello",
                "chat alpha -p Hello -s Brief --stats --json",
                "server start",
                "server start --port 8080 --bind 127.0.0.1 --cors --api-key k",
                "server stop",
                "server status --json",
                "log stream",
                "log stream --source model --filter input --stats",
                "daemon up",
                "daemon down",
                "daemon status --json",
                "runtime survey --json",
                "import $MODEL_FILE --dry-run",
                "--yes import $MODEL_FILE --copy --user-repo acme/new",
                "get acme/alpha",
                "--yes get acme/alpha@Q4_K_M --json",
                "completion zsh",
                "--backend lmstudio ls",
                "--backend lmstudio load alpha",
                "--host 127.0.0.1 --port 41234 --token secret-token status",
                "--quiet --yes --log-level debug ps",
            )

        /** How much of a failed case's output the failure tells. */
        const val TOLD = 1_500

        val VARIABLES = listOf("RUNA_HOST", "RUNA_PORT", "RUNA_BACKEND", "RUNA_LOG_LEVEL", "RUNA_LOG_PROMPTS", "RUNA_TOKEN", "LM_API_TOKEN")

        // What the options and the arguments take, right and wrong: numbers, durations, names of
        // models, keys of the configuration, levels, shells, addresses. No word here or in ODD is
        // the path of something outside the test's own directory: `import` moves what it is given.
        val VALUES =
            listOf("0", "1", "-1", "2", "4096", "65535", "65536", "99999999999999999999", "1.5", "-0.5", "NaN", "1e309") +
                listOf("max", "off", "auto", "true", "false", "yes", "5m", "0s", "90", "-3h", "1x", "1h30m") +
                listOf(FakeDaemon.ALPHA, FakeDaemon.BETA, "acme/alpha", "alpha", "ALPHA", "alpha:2", "acme/tiny@Q4_K_M", "tiny") +
                listOf("nope", "a/b/c@d", "@", "a@", "@b", "https://huggingface.co/acme/alpha", "acme/alpha/file.gguf") +
                listOf("port", "host", "backend", "log_level", "no.such.key", "debug", "info", "loud", "runa", "lmstudio", "ollama") +
                listOf("bash", "zsh", "fish", "powershell", "127.0.0.1", "0.0.0.0", "::1", "localhost", "http://x", "*") +
                listOf("model", "runtime", "server", "Hello", FakeDaemon.FAILING, "", " ")

        // What nobody means to type and somebody types.
        val ODD =
            listOf("--", "-", "---", "--=", "-=", "=", "--port=", "-h", "--help", "-vvv", "--no-such-option", "-x", "-y1") +
                listOf("@nope", "\u0000", "🙂", "a b", "\n", "é", "--json=1", "--yes=maybe", "'", "\"", "\\", "%s", "{}") +
                listOf("x".repeat(5_000), "--${"y".repeat(300)}", "-9", "--version", "-v")
    }
}
