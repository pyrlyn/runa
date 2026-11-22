package runa.cli

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.testing.test
import com.github.ajalt.mordant.rendering.AnsiLevel
import com.github.ajalt.mordant.table.table
import com.github.ajalt.mordant.terminal.Terminal
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Phase 0 spike (PLAN.md): Clikt parses and Mordant renders tables on every native target.
class CliRenderingSpikeTest {
    private class Greet : CliktCommand(name = "greet") {
        val name by argument()

        override fun run() = echo("hello $name")
    }

    @Test
    fun clikParsesArguments() {
        assertEquals("hello runa\n", Greet().test("runa").stdout)
    }

    @Test
    fun mordantRendersTable() {
        val terminal = Terminal(ansiLevel = AnsiLevel.NONE, width = 80)
        val rendered =
            terminal.render(
                table {
                    header { row("model", "size") }
                    body { row("qwen3-0.6b", "0.5 GB") }
                },
            )
        assertTrue("qwen3-0.6b" in rendered && "size" in rendered, rendered)
    }
}
