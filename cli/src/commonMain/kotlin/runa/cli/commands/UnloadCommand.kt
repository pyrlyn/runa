package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.UsageError
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.optional
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.mordant.widgets.Text
import kotlinx.coroutines.runBlocking
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.DaemonConnector
import runa.cli.output.pick
import runa.client.DaemonClient
import runa.core.UserInputError
import runa.core.api.UnloadRequest
import runa.core.api.UnloadResponse

internal class UnloadCommand : CliktCommand(name = "unload") {
    private val globals by requireObject<Globals>()
    private val identifier by argument("id", help = "The identifier `runa ps` shows; not needed when one model is loaded").optional()
    private val all by option("--all", help = "Unload every model").flag()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String = "Unload a model from the daemon and free its memory"

    override fun run() {
        if (all && identifier != null) throw UsageError("Give an identifier or --all, not both")
        val response =
            runBlocking {
                // A daemon that does not run has nothing loaded, so there is no reason to start one.
                val daemon = DaemonConnector(globals).find()
                if (daemon == null) {
                    globals.log.info { "The runa daemon is not running, so no model is loaded" }
                    return@runBlocking UnloadResponse(emptyList())
                }
                daemon.use {
                    val request = if (all) UnloadRequest(all = true) else UnloadRequest(instanceId = identifier ?: choose(it.client))
                    if (request.all || request.instanceId != null) it.client.unload(request) else UnloadResponse(emptyList())
                }
            }
        response.unloaded.forEach { globals.log.info { "Unloaded $it" } }
        // Text output is the identifiers, one a line; with none, an empty line would be noise in a pipe.
        if (response.unloaded.isEmpty() && !globals.wantsJson(json)) return
        globals.renderer(json).render(response, UnloadResponse.serializer()) { Text(it.unloaded.joinToString("\n")) }
    }

    /** The one loaded model, or the one picked from a list; null when nothing is loaded. */
    private suspend fun choose(client: DaemonClient): String? {
        val loaded = client.models().models.flatMap { it.loadedInstances }
        if (loaded.isEmpty()) {
            globals.log.info { "No models are loaded" }
            return null
        }
        if (loaded.size == 1) return loaded.single().id
        if (globals.yes || globals.wantsJson(json) || !terminal.terminalInfo.inputInteractive) {
            throw UserInputError("Name the model to unload, or use --all: ${loaded.joinToString(", ") { it.id }}")
        }
        val chosen =
            terminal.pick("Model", listOf("ID", "MODEL"), loaded.map { listOf(it.id, it.model) }) ?: throw UserInputError("No model chosen")
        return loaded[chosen].id
    }
}
