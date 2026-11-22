package runa.daemon

import okio.Path
import okio.Path.Companion.toPath
import runa.core.ModelKey
import runa.core.ModelKind
import runa.store.StoreListing
import runa.store.StoredModel

internal val modelsDir: Path = "/models".toPath()

/** A store entry without a file behind it: the fake engine never opens one. */
internal fun stored(repo: String): StoredModel =
    StoredModel(
        key = ModelKey("acme", repo, "Q4_K_M"),
        files = listOf(modelsDir / "acme" / repo / "$repo-Q4_K_M.gguf"),
        sizeBytes = 1_000,
        kind = ModelKind.LLM,
        architecture = "llama",
        name = "Tiny $repo",
        parameterCount = 1_000,
        quantization = "Q4_K_M",
        contextLength = 2048,
    )

internal val listing: StoreListing = StoreListing(listOf(stored("alpha"), stored("beta")), emptyList())

internal const val ALPHA: String = "acme/alpha@Q4_K_M"
