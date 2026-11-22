package runa.engine.llama

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import runa.core.ChatMessage
import runa.core.ChatRole
import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.core.UserInputError
import runa.engine.FinishReason
import runa.engine.GenerationEvent
import runa.engine.GenerationRequest
import runa.engine.Prompt
import runa.engine.Sampling
import runa.engine.llama.cinterop.LLAMA_POOLING_TYPE_MEAN
import runa.engine.toCompletion
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * The checks of `:engine`'s EngineContractTest on a real model, plus what only a real model can
 * show. Everything that needs the model is skipped unless RUNA_TEST_MODEL names one.
 */
class LlamaEngineTest {
    private val greedy = Sampling(temperature = 0.0)

    private fun chat(
        text: String,
        maxTokens: Int? = 48,
        sampling: Sampling = greedy,
        stop: List<String> = emptyList(),
    ) = GenerationRequest(Prompt.Chat(listOf(ChatMessage(ChatRole.USER, text))), sampling, maxTokens, stop)

    @Test
    fun describesItself() {
        val info = testEngine.info
        assertEquals("llama.cpp", info.name)
        assertTrue(Regex("\\d+\\.\\d+\\.\\d+").matches(info.version), info.version)
        assertEquals("cpu", info.accelerators.first())
    }

    @Test
    fun rejectsFilesThatAreNotModels() {
        val missing = "/nonexistent/model.gguf".toPath()
        val garbage = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-not-a-model-${Random.nextLong().toULong()}.gguf"
        FileSystem.SYSTEM.write(garbage) { writeUtf8("not a model at all") }
        try {
            runBlocking {
                for (file in listOf(missing, garbage)) {
                    assertFailsWith<UserInputError> { testEngine.load(file, LoadConfig()) }
                    assertFailsWith<UserInputError> { testEngine.estimateMemory(file, LoadConfig()) }
                }
            }
        } finally {
            FileSystem.SYSTEM.delete(garbage)
        }
    }

    @Test
    fun loadReportsProgressAndHonoursTheConfig() =
        withTestModelFile { path ->
            val progress = mutableListOf<Float>()
            testEngine.load(path, LoadConfig(contextLength = 512), progress::add).use { model ->
                assertEquals(512, model.contextLength)
                assertTrue(!model.supportsEmbeddings)
            }
            assertEquals(1f, progress.last())
            assertEquals(progress.sorted(), progress)
            assertTrue(progress.all { it in 0f..1f }, "$progress")

            val small = testEngine.estimateMemory(path, LoadConfig(contextLength = 512))
            val large = testEngine.estimateMemory(path, LoadConfig(contextLength = 2048))
            assertEquals(size(path), small.weightsBytes)
            assertTrue(small.contextBytes > 0)
            assertEquals(4 * small.contextBytes, large.contextBytes)
        }

    @Test
    fun aLinkedModelWeighsWhatItsTargetDoes() =
        withTestModelFile { path ->
            val link = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-linked-model-${Random.nextLong().toULong()}.gguf"
            FileSystem.SYSTEM.createSymlink(link, path)
            try {
                assertEquals(testEngine.estimateMemory(path, LoadConfig()), testEngine.estimateMemory(link, LoadConfig()))
            } finally {
                FileSystem.SYSTEM.delete(link)
            }
        }

    @Test
    fun aContextLongerThanTheModelsIsLowered() =
        withTestModelFile { path ->
            val trained = testEngine.estimateMemory(path, LoadConfig())
            assertEquals(trained, testEngine.estimateMemory(path, LoadConfig(contextLength = Int.MAX_VALUE)))
        }

    @Test
    fun generationStreamsTextAndEndsWithExactlyOneFinished() =
        withTestModel { model ->
            val events = model.generate(chat("Say hello in five words.")).toList()
            val finished = assertIs<GenerationEvent.Finished>(events.last())
            assertEquals(1, events.count { it is GenerationEvent.Finished })
            val text = events.filterIsInstance<GenerationEvent.Text>().joinToString("") { it.text }
            assertTrue(text.isNotBlank(), "no text was generated")
            assertTrue(finished.stats.promptTokens > 0)
            assertTrue(finished.stats.completionTokens > 0)
            assertTrue(finished.stats.tokensPerSecond > 0)
        }

    @Test
    fun theSameSeedGivesTheSameText() =
        withTestModel { model ->
            val request = chat("Write one sentence about the sea.")
            val first = model.generate(request).toCompletion()
            assertEquals(first.text, model.generate(request).toCompletion().text)

            val seeded = chat("Write one sentence about the sea.", sampling = Sampling(temperature = 1.2, seed = 7))
            assertEquals(model.generate(seeded).toCompletion().text, model.generate(seeded).toCompletion().text)
        }

    @Test
    fun maxTokensAndStopStringsEndGeneration() =
        withTestModel { model ->
            val capped = model.generate(chat("Count from 1 to 30.", maxTokens = 3)).toCompletion()
            assertEquals(FinishReason.LENGTH, capped.reason)
            assertEquals(3, capped.stats.completionTokens)

            val whole = model.generate(chat("Count from 1 to 30.")).toCompletion().text
            assertTrue(whole.length >= 12, "too short to cut: '$whole'")
            // A stop string taken from the middle of what the model says without one.
            val stop = whole.substring(whole.length / 2, whole.length / 2 + 3)
            val stopped = model.generate(chat("Count from 1 to 30.", stop = listOf(stop))).toCompletion()
            assertEquals(whole.substringBefore(stop), stopped.text)
            assertEquals(FinishReason.STOP, stopped.reason)
        }

    @Test
    fun runsWithoutTheGpuAndWithPartOfIt() =
        withTestModelFile { path ->
            for (gpu in listOf(GpuOffload.Off, GpuOffload.Ratio(0.5))) {
                testEngine.load(path, LoadConfig(contextLength = 512, gpu = gpu, threads = 2)).use { model ->
                    val completion = model.generate(GenerationRequest(Prompt.Text("1, 2, 3, 4, 5,"), greedy, maxTokens = 8)).toCompletion()
                    assertContains(completion.text, "6", message = "gpu ${gpu.id}")
                }
            }
        }

    @Test
    fun aPlainPromptIsContinued() =
        withTestModel { model ->
            val completion = model.generate(GenerationRequest(Prompt.Text("1, 2, 3, 4, 5,"), greedy, maxTokens = 8)).toCompletion()
            assertContains(completion.text, "6")
        }

    @Test
    fun cancellingTheCollectorLeavesTheModelUsable() =
        withTestModel { model ->
            val flow = model.generate(chat("Count from 1 to 30.", maxTokens = null))
            assertIs<GenerationEvent.Text>(flow.first())
            // The cancelled generation must have let go of the context.
            assertEquals(FinishReason.LENGTH, model.generate(chat("Count from 1 to 30.", maxTokens = 2)).toCompletion().reason)
        }

    @Test
    fun aPromptThatDoesNotFitIsAnError() =
        withTestModel(LoadConfig(contextLength = 256)) { model ->
            val error = assertFailsWith<UserInputError> { model.generate(chat("word ".repeat(400))).toList() }
            assertContains(error.message.orEmpty(), "context of ${model.contextLength} tokens")
        }

    @Test
    fun aFullContextEndsGeneration() =
        withTestModel(LoadConfig(contextLength = 256)) { model ->
            val completion = model.generate(chat("Count from 1 to 500, one number per line.", maxTokens = null)).toCompletion()
            assertEquals(FinishReason.LENGTH, completion.reason)
            assertEquals(model.contextLength, completion.stats.promptTokens + completion.stats.completionTokens)
        }

    @Test
    fun tokenizes() =
        withTestModel { model ->
            val tokens = model.tokenize("Hello, world!")
            assertTrue(tokens.size in 1..13, "$tokens")
            assertEquals(emptyList(), model.tokenize(""))
            assertNotEquals(tokens, model.tokenize("Goodbye, world!"))
        }

    @Test
    fun aClosedModelRejectsWork() =
        withTestModelFile { path ->
            val model = testEngine.load(path, LoadConfig(contextLength = 256))
            model.close()
            model.close()
            assertFailsWith<IllegalStateException> { model.generate(chat("x")).toList() }
            assertFailsWith<IllegalStateException> { model.tokenize("x") }
            assertFailsWith<IllegalStateException> { model.embed(listOf("x")) }
        }

    @Test
    fun aChatModelDoesNotEmbed() =
        withTestModel { model ->
            assertFailsWith<UserInputError> { model.embed(listOf("x")) }
        }

    @Test
    fun embedsWithPooling() =
        withTestModelFile { path ->
            // No embedding model is at hand, so a chat model is loaded the way one would be.
            testEngine.load(path, LoadConfig(contextLength = 256), pooling = LLAMA_POOLING_TYPE_MEAN).use { model ->
                assertTrue(model.supportsEmbeddings)
                val (cat, sameCat, physics) = model.embed(listOf("A cat sat on the mat.", "A cat sat on the mat.", "Quantum field theory."))
                assertTrue(cat.size > 16)
                assertEquals(1.0, dot(cat, cat), 1e-3)
                assertEquals(1.0, dot(cat, sameCat), 1e-3)
                assertTrue(dot(cat, physics) < 0.99, "different texts gave the same vector")
                assertFailsWith<UserInputError> { model.generate(chat("x")).toList() }
            }
        }

    private fun dot(
        a: FloatArray,
        b: FloatArray,
    ): Double = a.indices.sumOf { a[it].toDouble() * b[it] }

    private fun size(path: Path): Long = checkNotNull(FileSystem.SYSTEM.metadata(path).size)
}
