package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.switch
import com.github.ajalt.mordant.terminal.prompt
import com.github.ajalt.mordant.widgets.Text
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import okio.Path.Companion.toPath
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.requireRunaBackend
import runa.core.ModelKey
import runa.core.UserInputError
import runa.store.ImportMode

@Serializable
internal data class ImportResult(
    val model: String,
    val mode: String,
    @SerialName("dry_run") val dryRun: Boolean,
    val files: List<ImportedFile>,
)

@Serializable
internal data class ImportedFile(
    val source: String,
    val destination: String,
)

internal class ImportCommand : CliktCommand(name = "import") {
    private val globals by requireObject<Globals>()
    private val file by argument("file", help = "A .gguf file; any part of a split model brings all of its parts")
    private val userRepo by option("--user-repo", metavar = "publisher/repo", help = "Name the model instead of deriving it from the file")
    private val mode by option(help = "Keep the original file and copy, hard link or symlink it instead")
        .switch("--copy" to ImportMode.COPY, "--hard-link" to ImportMode.HARD_LINK, "--symlink" to ImportMode.SYMLINK)
        .default(ImportMode.MOVE, defaultForHelp = "move")
    private val dryRun by option("--dry-run", help = "Show what would be done and change nothing").flag()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String = "Add a GGUF file to the model store"

    override fun run() {
        requireRunaBackend(globals)
        val store = globals.store
        val source = store.inspect(file.toPath())
        val key = userRepo?.let(::parseKey) ?: chooseKey(source.suggestedKey)
        val plan = store.planImport(source, key)
        if (!dryRun) store.import(plan, mode)

        val (id, done) = WORDS.getValue(mode)
        val count = if (plan.steps.size == 1) "1 file" else "${plan.steps.size} files"
        globals.log.info { if (dryRun) "Dry run: $count would be $done as ${key.model}" else "Imported ${key.model} ($count $done)" }
        val result = ImportResult(key.model, id, dryRun, plan.steps.map { ImportedFile(it.source.toString(), it.destination.toString()) })
        globals.renderer(json).render(result, ImportResult.serializer()) { Text(it.files.joinToString("\n") { file -> file.destination }) }
    }

    private fun parseKey(text: String): ModelKey =
        ModelKey.parse(text)?.takeIf { it.variant == null }
            ?: throw UserInputError("'$text' is not a model name: use publisher/repo with letters, digits, '.', '_' and '-'")

    /** Asks for the name when someone is there to answer; scripts, `--json` and `-y` take the suggestion. */
    private fun chooseKey(suggested: ModelKey): ModelKey {
        if (globals.yes || globals.wantsJson(json) || !terminal.terminalInfo.inputInteractive) return suggested
        val answer = terminal.prompt("Import as", default = suggested.model) ?: throw UserInputError("No model name given")
        return parseKey(answer.trim())
    }

    private companion object {
        // The mode's name in JSON output, and its past participle for messages.
        val WORDS: Map<ImportMode, Pair<String, String>> =
            mapOf(
                ImportMode.MOVE to ("move" to "moved"),
                ImportMode.COPY to ("copy" to "copied"),
                ImportMode.HARD_LINK to ("hard-link" to "hard linked"),
                ImportMode.SYMLINK to ("symlink" to "symlinked"),
            )
    }
}
