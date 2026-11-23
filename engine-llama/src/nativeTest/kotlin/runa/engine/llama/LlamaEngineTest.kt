package runa.engine.llama

import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import runa.core.ChatMessage
import runa.core.ChatRole
import runa.core.GpuOffload
import runa.core.LoadConfig
import runa.core.ToolCall
import runa.core.UserInputError
import runa.engine.FinishReason
import runa.engine.GenerationEvent
import runa.engine.GenerationRequest
import runa.engine.Prompt
import runa.engine.ResponseFormat
import runa.engine.Sampling
import runa.engine.Tool
import runa.engine.ToolChoice
import runa.engine.llama.cinterop.LLAMA_POOLING_TYPE_MEAN
import runa.engine.toCompletion
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
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

    private fun json(text: String): JsonObject = Json.parseToJsonElement(text).jsonObject

    private val weather =
        Tool(
            "get_weather",
            "Gets the current weather in a city",
            json("""{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"""),
        )
    private val clock = Tool("get_time", "Gets the current time in a city", weather.parameters)
    private val question = ChatMessage(ChatRole.USER, "What is the weather in Paris?")

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
    fun aRequiredToolCallIsAnnouncedAndHasArgumentsOfItsSchema() =
        withTestModel { model ->
            val prompt = Prompt.Chat(listOf(question), listOf(weather), ToolChoice.Required)
            val events = model.generate(GenerationRequest(prompt, greedy, maxTokens = 96)).toList()
            val completion = events.asFlow().toCompletion()
            assertEquals(FinishReason.TOOL_CALLS, completion.reason, "$events")
            val call = completion.toolCalls.single()
            assertEquals("get_weather", call.name)
            assertTrue(call.id.startsWith("call_"), call.id)
            assertIs<JsonPrimitive>(json(call.arguments)["city"], call.arguments)

            val started = events.indexOfFirst { it is GenerationEvent.ToolCallStarted }
            assertEquals(1, events.count { it is GenerationEvent.ToolCallStarted })
            assertTrue(started < events.indexOfFirst { it is GenerationEvent.ToolCallArguments }, "$events")
        }

    @Test
    fun aNamedToolIsTheOneCalled() =
        withTestModel { model ->
            // Either tool answers half of it; left alone, the model takes the first.
            val both = ChatMessage(ChatRole.USER, "What is the weather in Paris, and what time is it there?")
            val prompt = Prompt.Chat(listOf(both), listOf(weather, clock), ToolChoice.Named("get_time"))
            val completion = model.generate(GenerationRequest(prompt, greedy, maxTokens = 96)).toCompletion()
            assertEquals(listOf("get_time"), completion.toolCalls.map { it.name }, completion.text)

            val unknown = Prompt.Chat(listOf(question), listOf(weather), ToolChoice.Named("get_date"))
            assertFailsWith<UserInputError> { model.generate(GenerationRequest(unknown, greedy)).toList() }
        }

    @Test
    fun aToolsResultIsReadAndWithoutACallTheAnswerIsText() =
        withTestModel { model ->
            val messages =
                listOf(
                    question,
                    ChatMessage(ChatRole.ASSISTANT, "", toolCalls = listOf(ToolCall("call_1", "get_weather", """{"city":"Paris"}"""))),
                    ChatMessage(ChatRole.TOOL, """{"temperature_celsius":21,"sky":"clear"}""", toolCallId = "call_1"),
                )
            val completion =
                model
                    .generate(
                        GenerationRequest(Prompt.Chat(messages, listOf(weather)), greedy, maxTokens = 64),
                    ).toCompletion()
            assertEquals(emptyList(), completion.toolCalls)
            assertContains(completion.text, "21")
            assertEquals(FinishReason.STOP, completion.reason)

            val forbidden = Prompt.Chat(listOf(question), listOf(weather), ToolChoice.None)
            val answer = model.generate(GenerationRequest(forbidden, greedy, maxTokens = 32)).toCompletion()
            assertEquals(emptyList(), answer.toolCalls)
            assertTrue(answer.text.isNotBlank())
            assertFalse("get_weather" in answer.text, answer.text)
        }

    @Test
    fun aJsonFormatShapesTheAnswer() =
        withTestModel { model ->
            val ask = listOf(ChatMessage(ChatRole.USER, "What is the capital of France?"))
            val schema =
                json("""{"type":"object","properties":{"capital":{"type":"string"}},"required":["capital"],"additionalProperties":false}""")
            val shaped =
                model
                    .generate(
                        GenerationRequest(Prompt.Chat(ask, format = ResponseFormat.Json(schema)), greedy, 64),
                    ).toCompletion()
            assertEquals(setOf("capital"), json(shaped.text).keys, shaped.text)
            assertEquals(FinishReason.STOP, shaped.reason)

            // Without a schema the answer is still an object, which a small model may never close.
            val free = model.generate(GenerationRequest(Prompt.Chat(ask, format = ResponseFormat.Json()), greedy, 48)).toCompletion()
            assertTrue(free.text.trimStart().startsWith("{"), free.text)
            if (free.reason == FinishReason.STOP) json(free.text)
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
    fun renderShowsWhatTheModelReads() =
        withTestModel { model ->
            assertEquals("1, 2, 3", model.render(Prompt.Text("1, 2, 3")))
            val rendered = model.render(Prompt.Chat(listOf(ChatMessage(ChatRole.SYSTEM, "Be brief."), question)))
            assertContains(rendered, "Be brief.")
            assertContains(rendered, question.content)
            // The template put its own marks around the messages.
            assertTrue(rendered.length > "Be brief.".length + question.content.length, rendered)
            // Rendering leaves the model as it was: it still answers.
            assertTrue(
                model
                    .generate(chat("Say hello.", maxTokens = 4))
                    .toCompletion()
                    .stats.completionTokens > 0,
            )
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
