package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.optional
import com.github.ajalt.clikt.parameters.options.convert
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.int
import com.github.ajalt.clikt.parameters.types.long
import com.github.ajalt.clikt.parameters.types.restrictTo
import com.github.ajalt.mordant.widgets.Text
import kotlinx.coroutines.runBlocking
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.chooseModel
import runa.cli.daemon.describe
import runa.cli.daemon.load
import runa.cli.daemon.loadedNote
import runa.cli.daemon.modelBackend
import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse

/** ` with context 8192, gpu max` after an estimate that `--auto` chose the settings of. */
private fun LoadResponse.chosenNote(): String = auto?.let { " with context ${it.contextLength}, gpu ${it.gpu}" }.orEmpty()

internal class LoadCommand : CliktCommand(name = "load") {
    private val globals by requireObject<Globals>()
    private val model by argument("model", help = "A model from `runa ls`; without one, a list to choose from is shown").optional()
    private val identifier by option("--identifier", metavar = "id", help = "The name to address the loaded model by (default: its key)")
    private val contextLength by option(
        "-c",
        "--context-length",
        metavar = "n",
        help = "Tokens of context (default: context_length in config.json, else ${LoadConfig.DEFAULT_CONTEXT_LENGTH})",
    ).int().restrictTo(min = 1)
    private val gpu by option(
        "--gpu",
        metavar = "off|max|0..1",
        help = "How much of the model runs on the GPU (default: gpu in config.json)",
    ).convert { GpuOffload.parse(it) ?: fail("use off, max or a number from 0 to 1") }
    private val parallel by option("--parallel", metavar = "n", help = "How many requests the model serves at once (default: 1)")
        .int()
        .restrictTo(min = 1)
    private val ttl by option(
        "--ttl",
        metavar = "seconds",
        help = "Unload the model after this long without a request; 0 for never (default: ttl in config.json)",
    ).long().restrictTo(min = 0)
    private val auto by option(
        "--auto",
        help = "Let the daemon choose --gpu and --context-length by the memory of its machine; one that is given is kept",
    ).flag()
    private val estimateOnly by option("--estimate-only", help = "Print the memory the model would take and load nothing").flag()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String =
        """
        Load a model into the daemon

        The model stays in memory until `runa unload`, its --ttl or `runa daemon down`. The daemon is
        started when it does not run. A model can be loaded more than once: the second copy is called
        `key:2`, or what --identifier says. On success the identifier is printed.

        With --auto the model goes to the GPU as a whole when it fits there, and gets the longest
        context that still fits, up to 32768 tokens and to what it was trained with. A model too large
        for the GPU is split between the GPU and the CPU, or runs on the CPU. A fifth of the memory
        stays free, and what the loaded models hold counts. With --estimate-only it loads nothing and
        tells what it would choose.

        With --backend lmstudio the model is loaded into LM Studio, which takes --context-length and
        none of the other settings.
        """.trimIndent()

    override fun run() {
        val forProgram = globals.wantsJson(json)
        val response =
            runBlocking {
                globals.modelBackend().use { backend ->
                    val request =
                        LoadRequest(
                            model = model ?: terminal.chooseModel(backend.models().models, mayAsk = !globals.yes && !forProgram),
                            contextLength = contextLength,
                            gpu = gpu?.id,
                            parallel = parallel,
                            ttl = ttl,
                            identifier = identifier,
                            estimateOnly = estimateOnly,
                            auto = auto,
                        )
                    backend.load(terminal, request, showProgress = !forProgram)
                }
            }
        response.loadedNote()?.let { note -> globals.log.info { note } }
        globals.renderer(json).render(response, LoadResponse.serializer()) {
            Text(it.instance?.id ?: ("Estimated memory: ${it.estimate?.describe() ?: "unknown"}" + it.chosenNote()))
        }
    }
}
