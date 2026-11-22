package runa.daemon

import okio.Path
import okio.Path.Companion.toPath
import runa.core.ModelKey
import runa.core.ModelKind
import runa.store.StoreListing
import runa.store.StoredModel

internal val modelsDir: Path = "/models".toPath()

/** A store entry without a file behind it: the fake engine never opens one. */
internal fun stored(
    repo: String,
    variant: String = "Q4_K_M",
): StoredModel =
    StoredModel(
        key = ModelKey("acme", repo, variant),
        files = listOf(modelsDir / "acme" / repo / "$repo-$variant.gguf"),
        sizeBytes = 1_000,
        kind = ModelKind.LLM,
        architecture = "llama",
        name = "Tiny $repo",
        parameterCount = 1_000,
        quantization = variant,
        contextLength = 2048,
    )

internal val listing: StoreListing = StoreListing(listOf(stored("alpha"), stored("beta")), emptyList())

internal const val ALPHA: String = "acme/alpha@Q4_K_M"
