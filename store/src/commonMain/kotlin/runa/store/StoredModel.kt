package runa.store

import okio.Path
import runa.core.ModelKey
import runa.core.ModelKind

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
