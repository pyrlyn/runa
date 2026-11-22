package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.options.convert
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.double
import com.github.ajalt.clikt.parameters.types.int
import com.github.ajalt.clikt.parameters.types.long
import com.github.ajalt.clikt.parameters.types.restrictTo
import com.github.ajalt.mordant.widgets.Text
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import okio.Path
import okio.Path.Companion.toPath
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.requireRunaBackend
import runa.cli.output.hundredths
import runa.cli.output.statsSummary
import runa.config.ConfigKey
import runa.core.ChatMessage
import runa.core.ChatRole
import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.engine.GenerationEvent
import runa.engine.GenerationRequest
import runa.engine.GenerationStats
import runa.engine.Prompt
import runa.engine.Sampling
import runa.store.find
import kotlin.time.DurationUnit

@Serializable
internal data class RunResult(
    val model: String,
    val text: String,
    /** `stop`: the model ended its answer; `length`: `--max-tokens` or the context ran out. */
    @SerialName("finish_reason") val finishReason: String,
    val stats: RunStats,
)

@Serializable
internal data class RunStats(
    @SerialName("prompt_tokens") val promptTokens: Int,
    @SerialName("completion_tokens") val completionTokens: Int,
    @SerialName("time_to_first_token_ms") val timeToFirstTokenMs: Long,
    @SerialName("generation_time_ms") val generationTimeMs: Long,
    @SerialName("tokens_per_second") val tokensPerSecond: Double,
) {
    constructor(stats: GenerationStats) : this(
        stats.promptTokens,
        stats.completionTokens,
        stats.timeToFirstToken.inWholeMilliseconds,
        stats.generationTime.inWholeMilliseconds,
        hundredths(stats.tokensPerSecond),
    )
}

internal class RunCommand : CliktCommand(name = "run") {
    private val globals by requireObject<Globals>()
    private val model by argument("model", help = "A model from `runa ls`, or the path of a .gguf file")
    private val prompt by argument("prompt", help = "What to ask the model")
    private val system by option("-s", "--system", metavar = "text", help = "System prompt")
    private val maxTokens by option("--max-tokens", metavar = "n", help = "Stop after this many tokens").int().restrictTo(min = 1)
    private val temperature by option("--temp", metavar = "t", help = "Sampling temperature; 0 always takes the most likely token")
        .double()
        .restrictTo(min = 0.0)
        .default(Sampling().temperature)
    private val seed by option("--seed", metavar = "n", help = "Random seed; the same seed gives the same answer").long()
    private val gpu by option(
        "--gpu",
        metavar = "off|max|0..1",
        help = "How much of the model runs on the GPU (default: gpu in config.json)",
    ).convert { GpuOffload.parse(it) ?: fail("use off, max or a number from 0 to 1") }
    private val contextLength by option(
        "-c",
        "--context-length",
        metavar = "n",
        help = "Tokens of context (default: context_length in config.json, else ${LoadConfig.DEFAULT_CONTEXT_LENGTH})",
    ).int().restrictTo(min = 1)
    private val stats by option("--stats", help = "Print token counts, time to first token and tokens per second").flag()
    private val json by option("--json", help = "Print one JSON object when the answer is complete").flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String =
        """
        Ask a model one question

        The model is loaded in this process, answers and is unloaded again: no daemon is needed.
        The answer is printed as it is generated; a longer context than the model has is lowered to it.
        """.trimIndent()

    override fun run() {
        requireRunaBackend(globals)
        val (name, path) = resolve()
        val config = globals.config
        val load =
            LoadConfig(
                contextLength =
                    contextLength ?: config.resolve(ConfigKey.CONTEXT_LENGTH).value?.toInt() ?: LoadConfig.DEFAULT_CONTEXT_LENGTH,
                gpu = gpu ?: GpuOffload.parse(config.resolve(ConfigKey.GPU).value.orEmpty()) ?: GpuOffload.Max,
            )
        val messages = listOfNotNull(system?.let { ChatMessage(ChatRole.SYSTEM, it) }, ChatMessage(ChatRole.USER, prompt))
        val request = GenerationRequest(Prompt.Chat(messages), Sampling(temperature = temperature, seed = seed), maxTokens)

        val streaming = !globals.wantsJson(json)
        val text = StringBuilder()
        val engine = globals.engine
        globals.log.debug {
            "Loading $path with ${engine.info.name} ${engine.info.version} (${engine.info.accelerators.joinToString(
                ", ",
            )})"
        }
        val finished =
            runBlocking {
                engine.load(path, load).use { loaded ->
                    globals.log.debug { "Context of ${loaded.contextLength} tokens" }
                    var finished: GenerationEvent.Finished? = null
                    loaded.generate(request).collect { event ->
                        when (event) {
                            is GenerationEvent.Text -> {
                                text.append(event.text)
                                if (streaming) terminal.rawPrint(event.text)
                            }

                            // `runa run` gives the model no tools.
                            is GenerationEvent.ToolCallStarted, is GenerationEvent.ToolCallArguments -> {}

                            is GenerationEvent.Finished -> {
                                finished = event
                            }
                        }
                    }
                    checkNotNull(finished) { "The engine ended generation without a Finished event" }
                }
            }

        val result = RunResult(name, text.toString(), finished.reason.id, RunStats(finished.stats))
        if (streaming) {
            // The shell prompt must start on a line of its own.
            if (!text.endsWith('\n')) terminal.rawPrint("\n")
        } else {
            globals.renderer(json).render(result, RunResult.serializer()) { Text(it.text) }
        }
        if (stats) {
            val summary =
                with(result.stats) {
                    statsSummary(
                        promptTokens,
                        completionTokens,
                        result.finishReason,
                        timeToFirstTokenMs / MILLIS_PER_SECOND,
                        tokensPerSecond,
                    )
                }
            terminal.println(terminal.theme.muted(summary), stderr = true)
        }
    }

    /** The name to report and the file to load. */
    private fun resolve(): Pair<String, Path> {
        val file = model.toPath()
        if (model.endsWith(GGUF_EXTENSION, ignoreCase = true) && globals.environment.fileSystem.exists(file)) return model to file
        val store = globals.store
        return store.list().find(model, store.modelsDir).let { it.key.toString() to it.path }
    }

    private companion object {
        const val GGUF_EXTENSION = ".gguf"
        const val MILLIS_PER_SECOND = 1000.0
    }
}
