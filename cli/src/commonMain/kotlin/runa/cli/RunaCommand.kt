package runa.cli

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.parameters.options.versionOption

/** The root `runa` command. Version and commit come from the executable's generated BuildInfo. */
public class RunaCommand(
    version: String,
    gitSha: String,
) : CliktCommand(name = "runa") {
    init {
        versionOption(version, names = setOf("-v", "--version"), message = { "runa $it ($gitSha)" })
    }

    override val printHelpOnEmptyArgs: Boolean = true

    override fun help(context: Context): String = "Local models from the terminal."

    override fun run(): Unit = Unit
}
