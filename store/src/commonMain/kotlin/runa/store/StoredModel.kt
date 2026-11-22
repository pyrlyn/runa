package runa.store

import okio.Path
import runa.core.ModelKey
import runa.core.ModelKind
import runa.core.UserInputError

/** One runnable model in the store: a GGUF file, or the parts of a split one. */
public data class StoredModel(
    /** Always has a variant. */
    val key: ModelKey,
    /** One file, or the parts of a split model in order. */
    val files: List<Path>,
    val sizeBytes: Long,
    val kind: ModelKind,
    val architecture: String?,
    /** `general.name` from the file, for display. */
    val name: String?,
    val parameterCount: Long,
    val quantization: String?,
    val contextLength: Long?,
) {
    /** The file to hand to an engine; for a split model, its first part. */
    val path: Path get() = files.first()
}

/** True when [name] is this model's key, its `publisher/repo` or its repo alone; case does not matter. */
public fun StoredModel.isNamed(name: String): Boolean =
    listOf(key.toString(), key.model, key.repo).any { it.equals(name, ignoreCase = true) }

/** A file in the store that could not be listed, and why. */
public data class StoreProblem(
    val path: Path,
    val message: String,
)

public data class StoreListing(
    /** Sorted by key. */
    val models: List<StoredModel>,
    val problems: List<StoreProblem>,
)

/**
 * The one model that [name] means.
 *
 * @throws UserInputError when no model or more than one is called that.
 */
public fun StoreListing.find(
    name: String,
    modelsDir: Path,
): StoredModel {
    val matches = models.filter { it.isNamed(name) }
    return when (matches.size) {
        1 -> matches.single()

        0 -> throw UserInputError("No model matches '$name' in $modelsDir. `runa ls` lists the models, `runa get` downloads one")

        else -> throw UserInputError(
            "'$name' matches ${matches.size} models, name one of them: ${matches.joinToString(", ") { it.key.toString() }}",
        )
    }
}
