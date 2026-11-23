package runa.cli

import com.github.ajalt.clikt.core.BaseCliktCommand
import com.github.ajalt.clikt.core.CliktCommand

/** One command of the tree: the words that lead to it and its options, with whether each takes a value. */
internal class CommandNode(
    val path: List<String>,
    val options: List<Pair<String, Boolean>>,
    /** True for a command that the help does not list, and for what is under one. */
    val hidden: Boolean,
)

/**
 * Every command of `runa`, the root first. The tree is read from inside it, so a test that walks
 * it meets every new command and option without a change.
 */
internal fun TestCli.commandTree(): List<CommandNode> {
    val probe = Probe()
    run(PROBE, extraCommands = listOf(probe))
    return probe.nodes
}

private const val PROBE = "command-tree-probe"

private class Probe : CliktCommand(name = PROBE) {
    val nodes = mutableListOf<CommandNode>()

    override fun run() = collect(currentContext.findRoot().command, emptyList(), hidden = false)

    private fun collect(
        command: BaseCliktCommand<*>,
        path: List<String>,
        hidden: Boolean,
    ) {
        val options = command.registeredOptions().flatMap { option -> option.names.map { it to (option.nvalues.last > 0) } }
        nodes += CommandNode(path, options, hidden)
        for (child in command.registeredSubcommands()) {
            if (child.commandName != PROBE) collect(child, path + child.commandName, hidden || child.hiddenFromHelp)
        }
    }
}
