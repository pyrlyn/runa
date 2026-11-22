package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.UsageError
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.optional
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.optionalValue
import com.github.ajalt.clikt.parameters.options.switch
import com.github.ajalt.mordant.terminal.prompt
import com.github.ajalt.mordant.widgets.Text
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import okio.Path
import okio.Path.Companion.toPath
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.requireRunaBackend
import runa.core.ModelKey
import runa.core.UserInputError
import runa.gguf.isGgufFileName
import runa.platform.HostOs
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
    private val file by argument("file", help = "A .gguf file; any part of a split model brings all of its parts").optional()
    private val fromLmStudio by option(
        "--from-lmstudio",
        metavar = "dir",
        help = "Instead of one file: symlink every GGUF model of LM Studio's models folder (default: ~/.lmstudio/models)",
        // Without `=`, the word after the flag would be taken for the folder.
    ).optionalValue("", acceptsUnattachedValue = false)
    private val userRepo by option("--user-repo", metavar = "publisher/repo", help = "Name the model instead of deriving it from the file")
    private val mode by option(help = "Keep the original file and copy, hard link or symlink it instead")
        .switch("--copy" to ImportMode.COPY, "--hard-link" to ImportMode.HARD_LINK, "--symlink" to ImportMode.SYMLINK)
        .default(ImportMode.MOVE, defaultForHelp = "move")
    private val dryRun by option("--dry-run", help = "Show what would be done and change nothing").flag()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String =
        """
        Add a GGUF file to the model store

        With --from-lmstudio the models that LM Studio downloaded are added under the names they have
        there, as symlinks unless --copy or --hard-link says otherwise: nothing is downloaded twice,
        and LM Studio keeps its files. Models that are already in the store are left alone.
        """.trimIndent()

    override fun run() {
        requireRunaBackend(globals)
        fromLmStudio?.let { return importFromLmStudio(it) }
        val store = globals.store
        val source = store.inspect((file ?: throw UsageError("Name the .gguf file to import, or use --from-lmstudio")).toPath())
        val key = userRepo?.let(::parseKey) ?: chooseKey(source.suggestedKey)
        val plan = store.planImport(source, key)
        if (!dryRun) store.import(plan, mode)

        val (id, done) = WORDS.getValue(mode)
        val count = if (plan.steps.size == 1) "1 file" else "${plan.steps.size} files"
        globals.log.info { if (dryRun) "Dry run: $count would be $done as ${key.model}" else "Imported ${key.model} ($count $done)" }
        val result = ImportResult(key.model, id, dryRun, plan.steps.map { ImportedFile(it.source.toString(), it.destination.toString()) })
        globals.renderer(json).render(result, ImportResult.serializer()) { Text(it.files.joinToString("\n") { file -> file.destination }) }
    }

    /**
     * LM Studio keeps a model as `publisher/model/file.gguf` under its models folder, which is the
     * layout of runa's own store, so every model keeps its name. What is not GGUF (MLX) is not
     * runa's to run and is passed over.
     */
    private fun importFromLmStudio(folder: String) {
        if (file != null ||
            userRepo != null
        ) {
            throw UsageError("--from-lmstudio takes every model as it is: give no file and no --user-repo")
        }
        val fileSystem = globals.environment.fileSystem
        val root = if (folder.isEmpty()) lmStudioModels() else folder.toPath()
        if (fileSystem.metadataOrNull(root)?.isDirectory != true) {
            throw UserInputError("LM Studio's models folder is not at $root; name it with --from-lmstudio=<dir>")
        }
        val store = globals.store
        // Moving would take the models away from LM Studio.
        val linking = if (mode == ImportMode.MOVE) ImportMode.SYMLINK else mode
        val (id, done) = WORDS.getValue(linking)
        val taken = mutableSetOf<Path>()
        val results = mutableListOf<ImportResult>()
        var present = 0
        val repos = fileSystem.list(root).sorted().flatMap { fileSystem.listOrNull(it).orEmpty().sorted() }
        for (repo in repos) {
            // A projector is a part of a vision model, not a model.
            val files =
                fileSystem.listOrNull(repo).orEmpty().sorted().filter {
                    isGgufFileName(it.name) &&
                        !it.name.contains("mmproj", ignoreCase = true)
                }
            if (files.isEmpty()) continue
            val key = ModelKey.parse("${checkNotNull(repo.parent).name}/${repo.name}")
            if (key == null) {
                globals.log.warn { "Skipped $repo: runa cannot use that name" }
                continue
            }
            for (model in files) {
                val source =
                    try {
                        store.inspect(model)
                    } catch (e: UserInputError) {
                        globals.log.warn { "Skipped: ${e.message}" }
                        continue
                    }
                // Every part of a split model names all of them.
                if (!taken.addAll(source.files)) continue
                val plan =
                    try {
                        store.planImport(source, key)
                    } catch (e: UserInputError) {
                        globals.log.debug { "Left alone: ${e.message}" }
                        present++
                        continue
                    }
                if (!dryRun) store.import(plan, linking)
                results +=
                    ImportResult(key.model, id, dryRun, plan.steps.map { ImportedFile(it.source.toString(), it.destination.toString()) })
            }
        }
        val count = if (results.size == 1) "1 model" else "${results.size} models"
        val already = if (present == 0) "" else "; $present already in the store"
        globals.log.info {
            when {
                results.isEmpty() && present == 0 -> "There are no GGUF models in $root"
                results.isEmpty() -> "Every GGUF model in $root is already in the store"
                dryRun -> "Dry run: $count would be $done from $root$already"
                else -> "Imported $count from $root ($done)$already"
            }
        }
        if (results.isEmpty() && !globals.wantsJson(json)) return
        globals.renderer(json).render(results, ListSerializer(ImportResult.serializer())) { all ->
            Text(all.flatMap { it.files }.joinToString("\n") { it.destination })
        }
    }

    /** Where LM Studio keeps its models unless it was told otherwise in its settings. */
    private fun lmStudioModels(): Path {
        val environment = globals.environment
        val home =
            environment.env(if (environment.os == HostOs.WINDOWS) "USERPROFILE" else "HOME")?.takeIf { it.isNotBlank() }
                ?: throw UserInputError("Cannot find the home directory; name LM Studio's models folder with --from-lmstudio=<dir>")
        return home.toPath() / ".lmstudio" / "models"
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
