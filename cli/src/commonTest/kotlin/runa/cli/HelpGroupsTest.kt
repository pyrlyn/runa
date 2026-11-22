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
                Grouped("expose", CommandGroup.SERVE),
                Grouped("tune", CommandGroup.LOCAL_MODELS),
            )
        val help = TestCli().run("--help", extraCommands = commands).stdout
        val sections = help.substringAfter("Show this message and exit\n\n")
        assertEquals(
            """
            |Local models:
            |  get     Download a GGUF model from Hugging Face
            |  ls      List the models on this machine
            |  import  Add a GGUF file to the model store
            |  run     Ask a model one question
            |  chat    Talk to a model
            |  load    Load a model into the daemon
            |  unload  Unload a model and free its memory
            |  ps      List the loaded models
            |  tune    Does tune
            |
            |Serve:
            |  status  Show whether the daemon runs and what it has loaded
            |  server  Serve the OpenAI-compatible API to other programs
            |  log     Watch what the daemon and its models do
            |  daemon  Start and stop the background process that keeps models loaded
            |  expose  Does expose
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
