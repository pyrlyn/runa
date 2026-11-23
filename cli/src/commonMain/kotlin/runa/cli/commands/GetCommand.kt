package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.mordant.animation.progress.MultiProgressBarAnimation
import com.github.ajalt.mordant.animation.progress.addTask
import com.github.ajalt.mordant.animation.progress.advance
import com.github.ajalt.mordant.widgets.Text
import com.github.ajalt.mordant.widgets.progress.completed
import com.github.ajalt.mordant.widgets.progress.percentage
import com.github.ajalt.mordant.widgets.progress.progressBar
import com.github.ajalt.mordant.widgets.progress.progressBarLayout
import com.github.ajalt.mordant.widgets.progress.speed
import com.github.ajalt.mordant.widgets.progress.text
import com.github.ajalt.mordant.widgets.progress.timeRemaining
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import okio.Path
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.output.formatBytes
import runa.cli.output.pick
import runa.client.LmStudioBackend
import runa.client.LmStudioDownload
import runa.core.Backend
import runa.core.BackendError
import runa.core.UserInputError
import runa.hub.HubClient
import runa.hub.HubRepo
import runa.hub.HubRequest
import runa.hub.HubVariant
import runa.hub.defaultVariant
import runa.hub.fits
import runa.hub.variantsOf

@Serializable
internal data class GetResult(
    /** What `runa ls` calls the model; null when the store cannot read what was downloaded. */
    val key: String?,
    val repo: String,
    val revision: String,
    val quantization: String?,
    @SerialName("size_bytes") val sizeBytes: Long,
    /** False when every file was already in the store. */
    val downloaded: Boolean,
    val files: List<String>,
)

internal class GetCommand : CliktCommand(name = "get") {
    private val globals by requireObject<Globals>()
    private val model by argument("model", help = "owner/repo, owner/repo@quantization, or a https://huggingface.co/ URL")
    private val select by option("--select", help = "Choose the variant from a list instead of taking the default").flag()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String =
        """
        Download a GGUF model from Hugging Face

        Without @quantization, the largest quantized variant that fits this machine's memory is taken.
        Gated and private models need an access token in HF_TOKEN. An interrupted download continues
        where it stopped when the command is run again.

        With --backend lmstudio a running LM Studio downloads the model into its own folder: a name
        from its catalog, or owner/repo@quantization and URLs as above. It goes on when this command is
        interrupted.
        """.trimIndent()

    override fun run() {
        if (globals.backend == Backend.LMSTUDIO) return getWithLmStudio()
        val request =
            HubRequest.parse(model)
                ?: throw UserInputError(
                    "'$model' is not a Hugging Face model: use owner/repo, owner/repo@quantization or a https://huggingface.co/ URL",
                )
        val repo = request.repo
        val store = globals.store
        val directory = store.modelsDir / repo.owner / repo.name

        val (variant, files, fetched) =
            globals.hub().use { hub ->
                runBlocking {
                    val variant = choose(request, variantsOf(hub.listFiles(repo)))
                    val (files, fetched) = download(hub, repo, variant, directory)
                    Triple(variant, files, fetched)
                }
            }

        // Reading it back through the store proves the files are a model, and gives the name `runa ls` shows.
        val listing = store.list()
        val names = files.map { it.name }
        val stored =
            listing.models.firstOrNull {
                it.key.publisher == repo.owner && it.key.repo == repo.name &&
                    it.files.map(Path::name) == names
            }
        if (stored == null) {
            val problem = listing.problems.firstOrNull { it.path.name in names }?.message ?: "it is not listed"
            globals.log.warn { "${names.first()} was downloaded, but the model store cannot use it: $problem" }
        }
        val shown = stored?.key?.toString() ?: "${repo.id}@${variant.label}"
        globals.log.info { if (fetched) "Downloaded $shown" else "$shown is already in the model store" }
        val result =
            GetResult(
                stored?.key?.toString(),
                repo.id,
                repo.revision,
                variant.quantization,
                variant.sizeBytes,
                fetched,
                files.map(Path::toString),
            )
        globals.renderer(json).render(result, GetResult.serializer()) { Text(it.files.joinToString("\n")) }
    }

    private fun choose(
        request: HubRequest,
        variants: List<HubVariant>,
    ): HubVariant {
        val repo = request.repo
        if (variants.isEmpty()) throw UserInputError("${repo.id} has no GGUF model files; look for a GGUF conversion of it on Hugging Face")
        val file = request.file
        val quantization = request.quantization
        val candidates =
            when {
                file != null -> {
                    variants
                        .filter { variant -> variant.files.any { it.path == file } }
                        .ifEmpty { throw UserInputError("${repo.id} has no GGUF model file $file") }
                }

                quantization != null -> {
                    variants
                        .filter { it.label.equals(quantization, ignoreCase = true) }
                        .ifEmpty {
                            throw UserInputError(
                                "${repo.id} has no $quantization variant; it has: ${variants.joinToString(", ") { it.label }}",
                            )
                        }
                }

                else -> {
                    variants
                }
            }
        val memory = globals.environment.physicalMemory()
        val default = checkNotNull(defaultVariant(candidates, memory))
        val chosen =
            when {
                candidates.size == 1 -> {
                    default
                }

                // Nobody to ask: scripts, `--json` and `-y` take the default.
                !select || globals.yes || globals.wantsJson(json) || !terminal.terminalInfo.inputInteractive -> {
                    globals.log.info { "${repo.id} has ${candidates.size} variants; taking ${default.label} (--select lists them)" }
                    default
                }

                else -> {
                    pick(candidates, default, memory)
                }
            }
        if (memory != null && !fits(chosen, memory)) {
            globals.log.warn {
                "${chosen.label} is ${formatBytes(chosen.sizeBytes)}; this machine has ${formatBytes(memory)} of memory, so it may not load"
            }
        }
        return chosen
    }

    private fun pick(
        candidates: List<HubVariant>,
        default: HubVariant,
        memory: Long?,
    ): HubVariant {
        val rows =
            candidates.map { variant ->
                val note =
                    when {
                        variant === default -> "default"
                        memory != null && !fits(variant, memory) -> "may not fit in memory"
                        else -> ""
                    }
                listOf(variant.label, formatBytes(variant.sizeBytes), note)
            }
        val chosen =
            terminal.pick("Variant", listOf("VARIANT", "SIZE", ""), rows, candidates.indexOf(default))
                ?: throw UserInputError("No variant chosen")
        return candidates[chosen]
    }

    private suspend fun download(
        hub: HubClient,
        repo: HubRepo,
        variant: HubVariant,
        directory: Path,
    ): Pair<List<Path>, Boolean> {
        val fileSystem = globals.environment.fileSystem
        val paths = variant.files.map { directory / it.name }
        // Checked here as well as by the client, so that nothing announces a download that will not happen.
        if (variant.files.zip(paths).all { (file, path) -> fileSystem.metadataOrNull(path)?.size == file.size }) return paths to false
        val parts = if (variant.files.size == 1) "" else " in ${variant.files.size} files"
        globals.log.info { "Downloading ${repo.id}@${variant.label} (${formatBytes(variant.sizeBytes)}$parts) to $directory" }
        if (globals.wantsJson(json) || !terminal.terminalInfo.outputInteractive) {
            return hub.download(repo, variant, directory, fileSystem)
        }

        // Drawn by hand instead of with Mordant's animator: that one hides the cursor, and Ctrl-C
        // ends the process too abruptly to show it again.
        val animation = MultiProgressBarAnimation(terminal)
        val task = animation.addTask(downloadLayout(variant.label), total = variant.sizeBytes)
        return coroutineScope {
            val redraw =
                launch {
                    while (true) {
                        animation.refresh()
                        delay(REDRAW_MILLIS)
                    }
                }
            try {
                val result = hub.download(repo, variant, directory, fileSystem) { task.advance(it) }
                // Nothing was fetched, so there is no progress worth leaving on the screen.
                if (result.second) animation.refresh(refreshAll = true) else animation.clear()
                result
            } finally {
                redraw.cancel()
            }
        }
    }

    private fun downloadLayout(label: String) =
        progressBarLayout {
            text(label)
            percentage()
            progressBar()
            completed(suffix = "B", includeTotal = true)
            speed("B/s")
            timeRemaining()
        }

    /** LM Studio downloads, into its own folder; this command starts the job and watches it. */
    private fun getWithLmStudio() {
        if (select) throw UserInputError("--select is not supported by the lmstudio backend: name the variant as owner/repo@quantization")
        // LM Studio takes a quantization only beside a Hugging Face link, so owner/repo@quantization becomes one.
        val quantization = if ('@' in model) model.substringAfterLast('@') else null
        val name = model.substringBeforeLast('@')
        val source = if (quantization != null && "://" !in name) "$HUGGING_FACE/$name" else name
        val job =
            runBlocking {
                globals.lmStudio().use { lmStudio ->
                    val started = lmStudio.download(source, quantization)
                    val id = started.jobId
                    if (started.status != LmStudioDownload.DOWNLOADING || id == null) started else follow(lmStudio, id, started)
                }
            }
        when (job.status) {
            LmStudioDownload.ALREADY_DOWNLOADED -> {
                globals.log.info { "$model is already in LM Studio" }
            }

            LmStudioDownload.COMPLETED -> {
                globals.log.info { "LM Studio downloaded $model" }
            }

            LmStudioDownload.PAUSED -> {
                globals.log.warn {
                    "The download of $model is paused in LM Studio and goes on when it is resumed there"
                }
            }

            else -> {
                throw BackendError("LM Studio could not download $model: the download is ${job.status}")
            }
        }
        globals.renderer(json).render(job, LmStudioDownload.serializer()) { Text(it.status) }
    }

    /** Asks about the job until it is no longer downloading, and draws how far it is when someone is watching. */
    private suspend fun follow(
        lmStudio: LmStudioBackend,
        id: String,
        started: LmStudioDownload,
    ): LmStudioDownload {
        val total = started.totalSizeBytes
        globals.log.info { "LM Studio downloads $model" + total?.let { " (${formatBytes(it)})" }.orEmpty() }
        val watched = total != null && !globals.wantsJson(json) && terminal.terminalInfo.outputInteractive
        val animation = MultiProgressBarAnimation(terminal)
        val task = if (watched) animation.addTask(downloadLayout(model), total = checkNotNull(total)) else null
        var drawn = 0L
        var job = started
        while (job.status == LmStudioDownload.DOWNLOADING) {
            delay(globals.environment.downloadPoll)
            job = lmStudio.downloadStatus(id)
            val now = job.downloadedBytes
            if (task != null && now != null) {
                task.advance(now - drawn)
                drawn = now
                animation.refresh()
            }
        }
        // What stays on the screen is the bar of a download that finished; any other is said in words.
        if (task != null && job.status != LmStudioDownload.COMPLETED) animation.clear()
        return job
    }

    private companion object {
        const val REDRAW_MILLIS = 100L
        const val HUGGING_FACE = "https://huggingface.co"
    }
}
