package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.UsageError
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.optional
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.mordant.rendering.Widget
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.output.formatBytes
import runa.cli.output.formatCount
import runa.cli.output.listing
import runa.core.Backend
import runa.core.ModelKey
import runa.core.ModelKind
import runa.core.UserInputError
import runa.core.api.ModelDescription
import runa.store.StoredModel
import runa.store.isNamed

/** One model file (or split set) in `runa ls --json`; the array has one of these per variant. */
@Serializable
internal data class ModelEntry(
    val key: String,
    val model: String,
    val variant: String,
    val type: String,
    val name: String?,
    val architecture: String?,
    val params: Long,
    val quantization: String?,
    @SerialName("context_length") val contextLength: Long?,
    @SerialName("size_bytes") val sizeBytes: Long,
    val path: String,
    val files: List<String>,
) {
    constructor(model: StoredModel) : this(
        key = model.key.toString(),
        model = model.key.model,
        variant = model.key.variant.orEmpty(),
        type = model.kind.id,
        name = model.name,
        architecture = model.architecture,
        params = model.parameterCount,
        quantization = model.quantization,
        contextLength = model.contextLength,
        sizeBytes = model.sizeBytes,
        path = model.path.toString(),
        files = model.files.map { it.toString() },
    )
}

internal class LsCommand : CliktCommand(name = "ls") {
    private val globals by requireObject<Globals>()
    private val key by argument("key", help = "Show the variants of one model, for example publisher/repo").optional()
    private val llm by option("--llm", help = "Show language models only").flag()
    private val embedding by option("--embedding", help = "Show embedding models only").flag()
    private val variants by option("--variants", help = "Show one row per file instead of one per model").flag()
    private val json by option("--json", help = JSON_HELP).flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String =
        """
        List the models on this machine

        With --backend lmstudio: the models of a running LM Studio, one row each, as its API lists them.
        """.trimIndent()

    override fun run() {
        if (llm && embedding) throw UsageError("--llm and --embedding cannot be used together")
        if (globals.backend == Backend.LMSTUDIO) return listLmStudio()
        val store = globals.store
        globals.log.debug { "Scanning ${store.modelsDir}" }
        val listing = store.list()
        listing.problems.forEach { globals.log.warn { "Skipped ${it.path}: ${it.message}" } }

        val kind = wantedKind()
        val wanted = key
        val models = listing.models.filter { model -> (kind == null || model.kind == kind) && (wanted == null || model.isNamed(wanted)) }
        if (wanted != null && models.isEmpty()) throw UserInputError("No model matches '$key' in ${store.modelsDir}")
        if (models.isEmpty() && !globals.wantsJson(json)) {
            globals.log.info { "No models in ${store.modelsDir}. Add one with: runa import <file.gguf>" }
            return
        }
        globals.renderer(json).render(models.map(::ModelEntry), ListSerializer(ModelEntry.serializer())) { entries ->
            if (variants || wanted != null) byVariant(entries) else byModel(entries)
        }
    }

    private fun wantedKind(): ModelKind? =
        when {
            llm -> ModelKind.LLM
            embedding -> ModelKind.EMBEDDING
            else -> null
        }

    /** What LM Studio lists, in the types of runa's own API: there is no file on this machine to describe. */
    private fun listLmStudio() {
        if (variants) throw UserInputError("--variants is not supported by the lmstudio backend: LM Studio lists every model once")
        val kind = wantedKind()?.id
        val wanted = key
        val models =
            runBlocking { globals.lmStudio().use { it.models().models } }.filter { model ->
                (kind == null || model.type == kind) &&
                    (wanted == null || model.key.equals(wanted, ignoreCase = true) || ModelKey.parse(model.key)?.isNamed(wanted) == true)
            }
        if (wanted != null && models.isEmpty()) throw UserInputError("No model matches '$wanted' in LM Studio")
        if (models.isEmpty() && !globals.wantsJson(json)) {
            globals.log.info { "LM Studio has no models. Download one with: runa --backend lmstudio get <model>" }
            return
        }
        globals.renderer(json).render(models, ListSerializer(ModelDescription.serializer())) { rows ->
            keyListing(
                rows.map {
                    listOf(
                        it.key,
                        it.type,
                        it.params?.let(::formatCount).orEmpty(),
                        it.architecture.orEmpty(),
                        it.quantization.orEmpty(),
                        formatBytes(it.sizeBytes),
                    )
                },
            )
        }
    }

    private fun keyListing(rows: List<List<String>>): Widget = listing(listOf("KEY", "TYPE", "PARAMS", "ARCH", "QUANT", "SIZE"), rows)

    private fun byVariant(entries: List<ModelEntry>): Widget =
        keyListing(
            entries.map {
                listOf(
                    it.key,
                    it.type,
                    formatCount(it.params),
                    it.architecture.orEmpty(),
                    it.quantization.orEmpty(),
                    formatBytes(it.sizeBytes),
                )
            },
        )

    private fun byModel(entries: List<ModelEntry>): Widget =
        listing(
            listOf("MODEL", "TYPE", "PARAMS", "ARCH", "VARIANTS", "SIZE"),
            entries.groupBy { it.model }.map { (model, files) ->
                val first = files.first()
                listOf(
                    model,
                    first.type,
                    formatCount(first.params),
                    first.architecture.orEmpty(),
                    files.joinToString(", ") { it.variant },
                    formatBytes(files.sumOf { it.sizeBytes }),
                )
            },
        )
}
