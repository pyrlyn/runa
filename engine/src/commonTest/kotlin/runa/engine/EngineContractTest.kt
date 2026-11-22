package runa.engine

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import okio.Path.Companion.toPath
import runa.core.ChatMessage
import runa.core.ChatRole
import runa.core.LoadConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

/** The behaviour every [Engine] must have, shown on [FakeEngine]; the llama.cpp engine runs the same checks in phase 4. */
class EngineContractTest {
    private val engine = FakeEngine()
    private val file = "/models/fake.gguf".toPath()

    private suspend fun load(config: LoadConfig = LoadConfig()) = engine.load(file, config) as FakeModel

    @Test
    fun loadReportsProgressAndHonoursTheConfig() =
        runTest {
            val progress = mutableListOf<Float>()
            val model = engine.load(file, LoadConfig(contextLength = 4), progress::add)
            assertEquals(listOf(0f, 1f), progress)
            assertEquals(4, model.contextLength)
            assertEquals(1_008, engine.estimateMemory(file, LoadConfig(contextLength = 4)).totalBytes)
        }

    @Test
    fun generationStreamsTextAndEndsWithExactlyOneFinished() =
        runTest {
            val events = load().generate(GenerationRequest(Prompt.Text("one two three"))).toList()
            assertEquals(listOf("one", " two", " three"), events.filterIsInstance<GenerationEvent.Text>().map { it.text })
            val finished = assertIs<GenerationEvent.Finished>(events.last())
            assertEquals(1, events.count { it is GenerationEvent.Finished })
            assertEquals(FinishReason.STOP, finished.reason)
            assertEquals(3, finished.stats.completionTokens)
        }

    @Test
    fun toCompletionJoinsTheStream() =
        runTest {
            val messages = listOf(ChatMessage(ChatRole.USER, "hi there"))
            val completion = load().generate(GenerationRequest(Prompt.Chat(messages))).toCompletion()
            assertEquals("user: hi there", completion.text)
            assertEquals(FinishReason.STOP, completion.reason)
        }

    @Test
    fun maxTokensAndStopStringsEndGeneration() =
        runTest {
            val model = load()
            val capped = model.generate(GenerationRequest(Prompt.Text("a b c d"), maxTokens = 2)).toCompletion()
            assertEquals("a b", capped.text)
            assertEquals(FinishReason.LENGTH, capped.reason)

            val stopped = model.generate(GenerationRequest(Prompt.Text("a b END c"), stop = listOf("END"))).toCompletion()
            assertEquals("a b", stopped.text)
            assertEquals(FinishReason.STOP, stopped.reason)
        }

    @Test
    fun cancellingTheCollectorStopsGeneration() =
        runTest {
            val model = load()
            val flow = model.generate(GenerationRequest(Prompt.Text("a b c d e")))
            assertEquals(0, model.generatedTokens, "a cold flow must not generate before it is collected")
            assertEquals(GenerationEvent.Text("a"), flow.first())
            assertEquals(2, flow.take(2).toList().size)
            assertTrue(model.generatedTokens <= 3, "generated ${model.generatedTokens} tokens after cancellation")
        }

    @Test
    fun aClosedModelRejectsWork() =
        runTest {
            val model = load()
            model.use { assertEquals(listOf(floatArrayOf(2f).toList()), it.embed(listOf("hi")).map(FloatArray::toList)) }
            assertTrue(model.closed)
            model.close()
            assertFailsWith<IllegalStateException> { model.generate(GenerationRequest(Prompt.Text("x"))).toList() }
            assertFailsWith<IllegalStateException> { model.embed(listOf("x")) }
        }

    @Test
    fun tokensPerSecondComesFromGenerationTime() {
        assertEquals(20.0, GenerationStats(5, 41, 100.milliseconds, 2.seconds).tokensPerSecond)
        assertEquals(0.0, GenerationStats(5, 1, 100.milliseconds, 0.seconds).tokensPerSecond)
    }
}
