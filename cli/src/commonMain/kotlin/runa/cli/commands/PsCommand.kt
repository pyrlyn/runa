package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.builtins.ListSerializer
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.modelBackend
import runa.cli.output.formatBytes
import runa.cli.output.formatSeconds
import runa.cli.output.listing
import runa.core.Backend
import runa.core.api.InstanceDescription

internal class PsCommand : CliktCommand(name = "ps") {
    private val globals by requireObject<Globals>()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String = "List the loaded models"

    override fun run() {
        val instances =
            runBlocking {
                globals.modelBackend().use { backend -> backend.models().models.flatMap { it.loadedInstances } }
            }
        if (instances.isEmpty() && !globals.wantsJson(json)) {
            globals.log.info { "No models are loaded. Load one with: runa load <model>" }
            return
        }
        // LM Studio's REST API tells neither the GPU share nor the idle time nor the TTL of an instance.
        val columns = if (globals.backend == Backend.LMSTUDIO) COLUMNS.filterNot { it.first in UNTOLD_BY_LM_STUDIO } else COLUMNS
        globals.renderer(json).render(instances, ListSerializer(InstanceDescription.serializer())) { rows ->
            listing(columns.map { it.first }, rows.map { row -> columns.map { it.second(row) } })
        }
    }

    private companion object {
        val COLUMNS: List<Pair<String, (InstanceDescription) -> String>> =
            listOf(
                "ID" to { it.id },
                "MODEL" to { it.model },
                "CONTEXT" to { it.contextLength.toString() },
                "GPU" to { it.gpu.orEmpty() },
                "PARALLEL" to { it.parallel?.toString().orEmpty() },
                "SIZE" to { formatBytes(it.sizeBytes) },
                "IDLE" to { it.idleSeconds?.let(::formatSeconds).orEmpty() },
                "TTL" to { it.ttlSeconds?.let(::formatSeconds) ?: "never" },
            )
        val UNTOLD_BY_LM_STUDIO = setOf("GPU", "IDLE", "TTL")
    }
}
