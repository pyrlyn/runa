package runa.cli

import runa.testing.assertGolden
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * `docs/commands.md` is the help of every command, and this test is what writes it
 * (`RUNA_UPDATE_GOLDEN=1`), so the reference cannot fall behind the commands.
 */
class CommandsReferenceTest {
    private val cli = TestCli()

    @Test
    fun theReferenceIsTheHelpOfEveryCommand() {
        val sections =
            cli.commandTree().filterNot { it.hidden }.map { node ->
                val run = cli.run(*node.path.toTypedArray(), "--help")
                assertEquals(0, run.exitCode, "runa ${node.path.joinToString(" ")} --help: ${run.stderr}")
                "## `${(listOf("runa") + node.path).joinToString(" ")}`\n\n```text\n${run.stdout.trimEnd()}\n```\n"
            }
        // Golden files live with the module's fixtures, and this one is the page people read.
        assertGolden("../../../../docs/commands.md", INTRODUCTION + sections.joinToString("\n"))
    }

    private companion object {
        val INTRODUCTION =
            """
            # Commands

            <!-- Written by CommandsReferenceTest from the help of every command. Do not edit it:
                 change the help, then run RUNA_UPDATE_GOLDEN=1 ./gradlew :cli:allTests. -->

            Every command of `runa` with the help it prints for `--help`. The options of `runa`
            itself go before the command (`runa --yes get <model>`), the options of a command after
            it. What the commands are for, in the order somebody new needs them, is in
            [getting-started.md](getting-started.md).

            A command ends with one of these exit codes:

            | Code | Meaning |
            | --- | --- |
            | 0 | Done |
            | 2 | A mistake in what was typed or asked for: an unknown option, a model that is not there |
            | 3 | The daemon, the hub or LM Studio did not answer or refused |
            | 1 | An error runa did not expect, with a stack trace: a bug to report |
            | 130 | Stopped with Ctrl-C |


            """.trimIndent()
    }
}
