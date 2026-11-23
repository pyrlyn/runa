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
import runa.cli.daemon.DaemonConnector
import runa.cli.output.formatBytes
import runa.cli.output.formatSeconds
import runa.cli.output.listing
import runa.core.api.InstanceDescription

internal class PsCommand : CliktCommand(name = "ps") {
    private val globals by requireObject<Globals>()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String = "List the models the daemon has loaded"

    override fun run() {
        val instances =
            runBlocking {
                DaemonConnector(globals).require().use { daemon ->
                    daemon.client
                        .models()
                        .models
                        .flatMap { it.loadedInstances }
                }
            }
        if (instances.isEmpty() && !globals.wantsJson(json)) {
            globals.log.info { "No models are loaded. Load one with: runa load <model>" }
            return
        }
        globals.renderer(json).render(instances, ListSerializer(InstanceDescription.serializer())) { rows ->
            listing(
                listOf("ID", "MODEL", "CONTEXT", "GPU", "PARALLEL", "SIZE", "IDLE", "TTL"),
                rows.map {
                    listOf(
                        it.id,
                        it.model,
                        it.contextLength.toString(),
                        it.gpu,
                        it.parallel.toString(),
                        formatBytes(it.sizeBytes),
                        formatSeconds(it.idleSeconds),
                        it.ttlSeconds?.let(::formatSeconds) ?: "never",
                    )
                },
            )
        }
    }
}
