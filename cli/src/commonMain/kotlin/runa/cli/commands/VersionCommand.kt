package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.mordant.widgets.Text
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import runa.cli.Globals

@Serializable
internal data class VersionInfo(
    val version: String,
    @SerialName("git_sha") val gitSha: String,
)

internal class VersionCommand : CliktCommand(name = "version") {
    private val globals by requireObject<Globals>()
    private val json by option("--json", help = JSON_HELP).flag()

    override fun help(context: Context): String = "Print the runa version"

    override fun run() {
        val info = VersionInfo(globals.environment.version, globals.environment.gitSha)
        globals.renderer(json).render(info, VersionInfo.serializer()) { Text("runa ${it.version} (${it.gitSha})") }
    }
}

internal const val JSON_HELP: String = "Print JSON"
