package runa.gguf

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GgufFileNameTest {
    @Test
    fun recognisesModelFiles() {
        assertTrue(isGgufFileName("model.gguf"))
        assertTrue(isGgufFileName("MODEL.GGUF"))
        assertFalse(isGgufFileName(".hidden.gguf"))
        assertFalse(isGgufFileName("model.gguf.part"))
        assertFalse(isGgufFileName("model.bin"))
    }

    @Test
    fun splitPartsShareAStem() {
        val part = GgufSplitPart.parse("big-Q4_K_M-00002-of-00003.gguf")
        assertEquals(GgufSplitPart("big-Q4_K_M", 2, 3), part)
        assertEquals("big-Q4_K_M-00001-of-00003.gguf", part?.fileName(1))
        assertNull(GgufSplitPart.parse("big-Q4_K_M.gguf"))
        assertEquals("big-Q4_K_M", ggufStem("big-Q4_K_M-00002-of-00003.gguf"))
        assertEquals("big-Q4_K_M", ggufStem("big-Q4_K_M.gguf"))
    }

    @Test
    fun findsTheQuantizationAtTheEndOfAName() {
        val cases =
            mapOf(
                "Qwen2.5-7B-Instruct-Q4_K_M" to "Q4_K_M",
                "model.IQ3_XXS" to "IQ3_XXS",
                "model-f16" to "F16",
                "model_bf16" to "BF16",
                "model-q8_0" to "Q8_0",
                "model-TQ1_0" to "TQ1_0",
                "Q4-is-in-the-middle-model" to null,
                "model-v2" to null,
                "plain" to null,
            )
        for ((stem, quantization) in cases) assertEquals(quantization, quantizationInName(stem), stem)
        assertEquals("Qwen2.5-7B-Instruct", withoutQuantization("Qwen2.5-7B-Instruct-Q4_K_M"))
        assertEquals("model-v2", withoutQuantization("model-v2"))
    }
}
