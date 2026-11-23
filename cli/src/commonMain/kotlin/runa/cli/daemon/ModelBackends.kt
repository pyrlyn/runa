package runa.cli.daemon

import com.github.ajalt.clikt.core.CliktCommand
import runa.cli.Globals
import runa.client.ModelBackend
import runa.core.Backend
import runa.core.UserInputError

// Where the commands that work with either backend get the service they talk to, and where the
// others turn the LM Studio backend down.

/**
 * The service that holds the models; the caller closes it. runa's own daemon is started when it
 * does not run. LM Studio is never started: that it is missing shows on the first request.
 */
internal suspend fun Globals.modelBackend(): ModelBackend =
    when (backend) {
        Backend.RUNA -> DaemonConnector(this).require().client
        Backend.LMSTUDIO -> lmStudio()
    }

/** As [modelBackend], but a daemon that does not run stays that way: null then. */
internal suspend fun Globals.runningModelBackend(): ModelBackend? =
    when (backend) {
        Backend.RUNA -> DaemonConnector(this).find()?.client
        Backend.LMSTUDIO -> lmStudio()
    }

/** Ends a command that needs runa's own daemon, engine or model store when another backend is chosen. */
internal fun CliktCommand.requireRunaBackend(globals: Globals) {
    val backend = globals.backend
    if (backend == Backend.RUNA) return
    val command = currentContext.commandNameWithParents().joinToString(" ")
    throw UserInputError(
        "`$command` is not supported by the ${backend.id} backend; `--backend ${Backend.RUNA.id}` runs it with runa itself",
    )
}
