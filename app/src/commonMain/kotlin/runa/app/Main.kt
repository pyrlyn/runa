package runa.app

import runa.cli.CliEnvironment
import runa.cli.runCli
import runa.core.ExitCode
import runa.platform.exitOnInterrupt
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    exitOnInterrupt(ExitCode.INTERRUPTED)
    exitProcess(runCli(args.asList(), CliEnvironment(BuildInfo.VERSION, BuildInfo.GIT_SHA)))
}
