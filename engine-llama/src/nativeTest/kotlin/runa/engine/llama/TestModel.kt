package runa.engine.llama

import kotlinx.coroutines.runBlocking
import okio.Path
import okio.Path.Companion.toPath
import runa.core.LoadConfig
import runa.engine.LoadedModel
import runa.platform.systemEnv

/** The variable that names a small chat model in GGUF format; tests that need a real model skip without it. */
internal const val TEST_MODEL_VARIABLE = "RUNA_TEST_MODEL"

internal val testEngine = LlamaEngine()

/** Runs [test] with the model's path, or reports the test as skipped when the variable is unset. */
internal fun withTestModelFile(test: suspend (Path) -> Unit) {
    val path = systemEnv(TEST_MODEL_VARIABLE)?.takeIf { it.isNotBlank() }
    if (path == null) {
        println("Skipped: set $TEST_MODEL_VARIABLE to the path of a small GGUF chat model")
        return
    }
    runBlocking { test(path.toPath()) }
}

internal fun withTestModel(
    config: LoadConfig = LoadConfig(contextLength = 1024),
    test: suspend (LoadedModel) -> Unit,
) = withTestModelFile { path -> testEngine.load(path, config).use { test(it) } }
