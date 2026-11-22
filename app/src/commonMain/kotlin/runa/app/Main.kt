package runa.app

import com.github.ajalt.clikt.core.main
import runa.cli.RunaCommand

fun main(args: Array<String>) {
    RunaCommand(version = BuildInfo.VERSION, gitSha = BuildInfo.GIT_SHA).main(args)
}
