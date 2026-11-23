package runa.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ModelKeyTest {
    @Test
    fun parsesKeysWithAndWithoutAVariant() {
        assertEquals(ModelKey("Qwen", "Qwen2.5-7B-Instruct-GGUF"), ModelKey.parse("Qwen/Qwen2.5-7B-Instruct-GGUF"))
        assertEquals(ModelKey("local", "tiny", "q4_k_m"), ModelKey.parse("local/tiny@q4_k_m"))
        assertEquals("local/tiny@q4_k_m", ModelKey("local", "tiny", "q4_k_m").toString())
        assertEquals("local/tiny", ModelKey("local", "tiny", "q4_k_m").model)
    }

    @Test
    fun rejectsAnythingThatCouldLeaveTheModelsDir() {
        val invalid =
            listOf(
                "",
                "tiny",
                "a/b/c",
                "../etc",
                "a/..",
                "a/.hidden",
                "a/b..c",
                "a b/c",
                "a/b@",
                "a/b@../x",
                "/a/b",
                "a\\b/c",
                "a/" + "x".repeat(97),
            )
        for (text in invalid) {
            assertNull(ModelKey.parse(text), "input: '$text'")
        }
    }

    @Test
    fun toValidNameCleansFreeText() {
        assertEquals("Meta-Llama", ModelKey.toValidName("Meta Llama"))
        assertEquals("model.v2", ModelKey.toValidName("..model..v2.."))
        assertEquals("a-.-b", ModelKey.toValidName("a/../b"))
        assertNull(ModelKey.toValidName("///"))
        assertNull(ModelKey.toValidName(""))
    }
}
