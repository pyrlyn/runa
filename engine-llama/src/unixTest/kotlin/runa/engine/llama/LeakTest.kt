package runa.engine.llama

import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import platform.posix.RUSAGE_SELF
import platform.posix.getrusage
import platform.posix.rusage
import runa.core.ChatMessage
import runa.core.ChatRole
import runa.core.LoadConfig
import runa.engine.GenerationRequest
import runa.engine.Prompt
import runa.engine.Sampling
import runa.engine.toCompletion
import kotlin.test.Test
import kotlin.test.assertTrue

class LeakTest {
    /** PLAN.md, phase 4: load and unload 50 times, memory stable within 5 %. */
    @Test
    fun loadingAndUnloadingDoesNotGrowTheProcess() =
        withTestModelFile { path ->
            val request =
                GenerationRequest(Prompt.Chat(listOf(ChatMessage(ChatRole.USER, "Hi"))), Sampling(temperature = 0.0), maxTokens = 4)

            suspend fun cycle() = testEngine.load(path, LoadConfig(contextLength = 512)).use { it.generate(request).toCompletion() }

            // The first cycles pay for what is allocated once per process (Metal, the thread pool).
            repeat(WARM_UP) { cycle() }
            val before = peakResidentSize()
            repeat(CYCLES) { cycle() }
            val after = peakResidentSize()
            // The peak can only grow, and a leak makes it grow with every cycle.
            assertTrue(after <= before * 1.05, "peak resident size grew from $before to $after over $CYCLES load cycles")
        }

    // In bytes on macOS and kilobytes on Linux: only the ratio is used.
    private fun peakResidentSize(): Long =
        memScoped {
            val usage = alloc<rusage>()
            check(getrusage(RUSAGE_SELF, usage.ptr) == 0) { "getrusage failed" }
            usage.ru_maxrss
        }

    private companion object {
        const val WARM_UP = 5
        const val CYCLES = 50
    }
}
