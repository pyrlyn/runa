package runa.cli

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class HelpGroupsTest {
    private class Grouped(
        name: String,
        group: CommandGroup,
    ) : CliktCommand(name) {
        override val helpTags: Map<String, String> = group.helpTags

        override fun help(context: Context): String = "Does $commandName"

        override fun run() = Unit
    }

    @Test
    fun commandsAreListedUnderTheirGroupsInOrder() {
        // Registered out of order on purpose: the layout must not depend on it.
        val commands =
            listOf(
                Grouped("survey", CommandGroup.RUNTIME),
                Grouped("server", CommandGroup.SERVE),
                Grouped("ls", CommandGroup.LOCAL_MODELS),
                Grouped("load", CommandGroup.LOCAL_MODELS),
            )
        val help = TestCli().run("--help", extraCommands = commands).stdout
        val sections = help.substringAfter("Show this message and exit\n\n")
        assertEquals(
            """
            |Local models:
            |  ls    Does ls
            |  load  Does load
            |
            |Serve:
            |  server  Does server
            |
            |Runtime:
            |  survey  Does survey
            |
            |Commands:
            |  version     Print the runa version
            |  config      Show and change settings (config.json)
            |  completion  Print a completion script for bash, zsh or fish
            |
            """.trimMargin(),
            sections,
        )
        assertFalse(CommandGroup.TAG in help)
    }
}
