package runa.cli

import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.output.HelpFormatter.ParameterHelp
import com.github.ajalt.clikt.output.MordantHelpFormatter
import com.github.ajalt.mordant.rendering.Widget

/** The sections of `runa --help`, in display order, named as in `lms`. */
internal enum class CommandGroup(
    val title: String,
) {
    LOCAL_MODELS("Local models"),
    SERVE("Serve"),
    RUNTIME("Runtime"),
    ;

    /** Put this in a command's `helpTags` to list it under the group. */
    val helpTags: Map<String, String> get() = mapOf(TAG to name)

    companion object {
        const val TAG: String = "runa.group"
    }
}

/** Mordant's help layout, with subcommands split into [CommandGroup] sections; ungrouped ones come last. */
internal class GroupedHelpFormatter(
    context: Context,
) : MordantHelpFormatter(context, showDefaultValues = true) {
    override fun renderCommands(parameters: List<ParameterHelp>): List<RenderedSection<Widget>> {
        val byGroup =
            parameters
                .filterIsInstance<ParameterHelp.Subcommand>()
                .groupBy { command -> CommandGroup.entries.firstOrNull { it.name == command.tags[CommandGroup.TAG] } }
        return (CommandGroup.entries + null).mapNotNull { group ->
            val commands = byGroup[group] ?: return@mapNotNull null
            val rows = commands.map { DefinitionRow(styleSubcommandName(it.name), renderParameterHelpText(it.help, it.tags)) }
            val title = group?.title ?: localization.commandsTitle()
            RenderedSection(styleSectionTitle(renderSectionTitle(title)), buildParameterList(rows))
        }
    }

    override fun shouldShowTag(
        tag: String,
        value: String,
    ): Boolean = tag != CommandGroup.TAG && super.shouldShowTag(tag, value)
}
