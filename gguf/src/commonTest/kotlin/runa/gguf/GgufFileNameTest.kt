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
    fun siblingNamesKeepTheExtensionAsTheFileSpellsIt() {
        assertEquals("Big-00002-of-00002.GGUF", GgufSplitPart.parse("Big-00001-of-00002.GGUF")?.fileName(2))
        assertEquals("big-00001-of-00002.Gguf", GgufSplitPart.parse("big-00002-of-00002.Gguf")?.fileName(1))
        assertEquals("Big", ggufStem("Big-00001-of-00002.GGUF"))
    }

    @Test
    fun aPartNumberOutsideItsTotalIsNotAValidPart() {
        assertTrue(GgufSplitPart.parse("big-00001-of-00001.gguf")!!.isValid)
        assertTrue(GgufSplitPart.parse("big-00002-of-00002.gguf")!!.isValid)
        for (name in listOf("big-00000-of-00000.gguf", "big-00001-of-00000.gguf", "big-00000-of-00002.gguf", "big-00003-of-00002.gguf")) {
            assertFalse(GgufSplitPart.parse(name)!!.isValid, name)
        }
    }

    @Test
    fun aSplitIsCompleteOnlyWithEveryPartFromOneToItsTotal() {
        assertTrue(GgufSplitPart.isComplete(listOf(1, 2, 3), 3))
        assertTrue(GgufSplitPart.isComplete(listOf(2, 1), 2))
        assertFalse(GgufSplitPart.isComplete(listOf(1, 3), 2), "the right count, a part missing")
        assertFalse(GgufSplitPart.isComplete(listOf(1, 1), 2), "the same part twice")
        assertFalse(GgufSplitPart.isComplete(listOf(1), 2))
        assertFalse(GgufSplitPart.isComplete(emptyList(), 0))
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
