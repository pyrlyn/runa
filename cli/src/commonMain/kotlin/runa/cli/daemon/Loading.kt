package runa.cli.daemon

import com.github.ajalt.mordant.animation.progress.MultiProgressBarAnimation
import com.github.ajalt.mordant.animation.progress.addTask
import com.github.ajalt.mordant.animation.progress.advance
import com.github.ajalt.mordant.terminal.Terminal
import com.github.ajalt.mordant.widgets.progress.percentage
import com.github.ajalt.mordant.widgets.progress.progressBar
import com.github.ajalt.mordant.widgets.progress.progressBarLayout
import com.github.ajalt.mordant.widgets.progress.text
import runa.cli.output.formatBytes
import runa.cli.output.pick
import runa.client.DaemonClient
import runa.core.UserInputError
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.MemoryEstimateDescription
import runa.core.api.ModelDescription
import kotlin.math.roundToLong

// What the commands that load a model into the daemon share: `load`, and `chat` for a model that is not loaded yet.

/**
 * The model to load when the command line names none: the only one there is, or the one the user
 * picks from a list. With nobody to ask ([mayAsk] is false, or no terminal), only a store with one
 * model has an answer.
 */
internal fun Terminal.chooseModel(
    models: List<ModelDescription>,
    mayAsk: Boolean,
): String {
    if (models.isEmpty()) throw UserInputError("There are no models to load; download one with `runa get`")
    if (models.size == 1) return models.single().key
    if (!mayAsk || !terminalInfo.inputInteractive) {
        throw UserInputError("Name the model to load: ${models.joinToString(", ") { it.key }}")
    }
    val rows = models.map { listOf(it.key, it.type, formatBytes(it.sizeBytes), if (it.loadedInstances.isEmpty()) "" else "loaded") }
    val chosen = pick("Model", listOf("KEY", "TYPE", "SIZE", ""), rows) ?: throw UserInputError("No model chosen")
    return models[chosen].key
}

/** Loads a model and returns when it is ready, with a progress bar meanwhile when [showProgress] and someone is watching. */
internal suspend fun DaemonClient.load(
    terminal: Terminal,
    request: LoadRequest,
    showProgress: Boolean,
): LoadResponse {
    if (!showProgress || request.estimateOnly || !terminal.terminalInfo.outputInteractive) return load(request)
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
        return load(request) { progress ->
            val now = (progress * PROGRESS_STEPS).toLong()
            task.advance(now - drawn)
            drawn = now
            animation.refresh()
        }
    } finally {
        animation.clear()
    }
}

/** `1.2 GB (weights 1.1 GB, context 100 MB)`. */
internal fun MemoryEstimateDescription.describe(): String =
    "${formatBytes(totalBytes)} (weights ${formatBytes(weightsBytes)}, context ${formatBytes(contextBytes)})"

/** The note about a model that was just loaded; null when the answer is only an estimate. */
internal fun LoadResponse.loadedNote(): String? {
    val instance = instance ?: return null
    val seconds = ((loadTimeSeconds ?: 0.0) * 100).roundToLong() / 100.0
    return "Loaded ${instance.model} in $seconds s: context ${instance.contextLength}, gpu ${instance.gpu}, about ${estimate.describe()}"
}

private const val PROGRESS_STEPS = 1_000L
