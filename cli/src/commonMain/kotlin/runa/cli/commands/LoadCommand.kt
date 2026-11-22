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
import com.github.ajalt.mordant.animation.progress.MultiProgressBarAnimation
import com.github.ajalt.mordant.animation.progress.addTask
import com.github.ajalt.mordant.animation.progress.advance
import com.github.ajalt.mordant.widgets.Text
import com.github.ajalt.mordant.widgets.progress.percentage
import com.github.ajalt.mordant.widgets.progress.progressBar
import com.github.ajalt.mordant.widgets.progress.progressBarLayout
import com.github.ajalt.mordant.widgets.progress.text
import kotlinx.coroutines.runBlocking
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.DaemonConnector
import runa.cli.output.formatBytes
import runa.cli.output.pick
import runa.client.DaemonClient
import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.core.UserInputError
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.ModelDescription
import kotlin.math.roundToLong

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
    private val estimateOnly by option("--estimate-only", help = "Print the memory the model would take and load nothing").flag()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String =
        """
        Load a model into the daemon

        The model stays in memory until `runa unload`, its --ttl or `runa daemon down`. The daemon is
        started when it does not run. A model can be loaded more than once: the second copy is called
        `key:2`, or what --identifier says. On success the identifier is printed.
        """.trimIndent()

    override fun run() {
        val response =
            runBlocking {
                DaemonConnector(globals).require().use { daemon ->
                    val request =
                        LoadRequest(
                            model = model ?: choose(daemon.client.models().models),
                            contextLength = contextLength,
                            gpu = gpu?.id,
                            parallel = parallel,
                            ttl = ttl,
                            identifier = identifier,
                            estimateOnly = estimateOnly,
                        )
                    load(daemon.client, request)
                }
            }
        val estimate = response.estimate
        val memory = "${formatBytes(
            estimate.totalBytes,
        )} (weights ${formatBytes(estimate.weightsBytes)}, context ${formatBytes(estimate.contextBytes)})"
        val instance = response.instance
        if (instance != null) {
            val seconds = ((response.loadTimeSeconds ?: 0.0) * 100).roundToLong() / 100.0
            globals.log.info {
                "Loaded ${instance.model} in $seconds s: context ${instance.contextLength}, gpu ${instance.gpu}, about $memory"
            }
        }
        globals.renderer(json).render(response, LoadResponse.serializer()) { Text(instance?.id ?: "Estimated memory: $memory") }
    }

    /** With nobody to ask, only a store with one model has an answer. */
    private fun choose(models: List<ModelDescription>): String {
        if (models.isEmpty()) throw UserInputError("There are no models to load; download one with `runa get`")
        if (models.size == 1) return models.single().key
        if (globals.yes || globals.wantsJson(json) || !terminal.terminalInfo.inputInteractive) {
            throw UserInputError("Name the model to load: ${models.joinToString(", ") { it.key }}")
        }
        val rows = models.map { listOf(it.key, it.type, formatBytes(it.sizeBytes), if (it.loadedInstances.isEmpty()) "" else "loaded") }
        val chosen = terminal.pick("Model", listOf("KEY", "TYPE", "SIZE", ""), rows) ?: throw UserInputError("No model chosen")
        return models[chosen].key
    }

    private suspend fun load(
        client: DaemonClient,
        request: LoadRequest,
    ): LoadResponse {
        if (request.estimateOnly || globals.wantsJson(json) || !terminal.terminalInfo.outputInteractive) return client.load(request)
        val layout =
            progressBarLayout {
                text(request.model)
                percentage()
                progressBar()
            }
        // Drawn by hand, as in `get`: Mordant's animator hides the cursor and Ctrl-C would leave it hidden.
        val animation = MultiProgressBarAnimation(terminal)
        val task = animation.addTask(layout, total = PROGRESS_STEPS)
        var drawn = 0L
        try {
            return client.load(request) { progress ->
                val now = (progress * PROGRESS_STEPS).toLong()
                task.advance(now - drawn)
                drawn = now
                animation.refresh()
            }
        } finally {
            animation.clear()
        }
    }

    private companion object {
        const val PROGRESS_STEPS = 1_000L
    }
}
