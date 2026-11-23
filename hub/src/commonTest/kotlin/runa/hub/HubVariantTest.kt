package runa.hub

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HubVariantTest {
    private fun file(
        path: String,
        size: Long,
    ) = HubFile(path, size, sha256 = null)

    private val gb = 1_000_000_000L

    private val files =
        listOf(
            file("README.md", 1_000),
            file("model-Q8_0.gguf", 8 * gb),
            file("model-Q4_K_M.gguf", 4 * gb),
            file("model-f16.gguf", 15 * gb),
            file("mmproj-model-f16.gguf", gb),
            file("big-Q6_K-00002-of-00002.gguf", 5 * gb),
            file("big-Q6_K-00001-of-00002.gguf", 6 * gb),
            file("broken-Q2_K-00001-of-00003.gguf", gb),
        )

    @Test
    fun groupsGgufFilesIntoVariantsSmallestFirst() {
        val variants = variantsOf(files)
        assertEquals(listOf("Q4_K_M", "Q8_0", "Q6_K", "F16"), variants.map { it.label })
        assertEquals(listOf(4 * gb, 8 * gb, 11 * gb, 15 * gb), variants.map { it.sizeBytes })
        assertEquals(
            listOf("big-Q6_K-00001-of-00002.gguf", "big-Q6_K-00002-of-00002.gguf"),
            variants[2].files.map { it.name },
        )
    }

    @Test
    fun aVariantWithoutQuantizationInItsNameIsLabelledByItsFileName() {
        val variant = variantsOf(listOf(file("tiny.gguf", 10))).single()
        assertNull(variant.quantization)
        assertEquals("tiny", variant.label)
    }

    @Test
    fun theDefaultIsTheLargestQuantizedVariantThatFits() {
        val variants = variantsOf(files)
        assertEquals("Q6_K", defaultVariant(variants, 16 * gb)?.label)
        assertEquals("Q8_0", defaultVariant(variants, 12 * gb)?.label)
        // Full precision is never the default while a quantized file exists, however much memory there is.
        assertEquals("Q6_K", defaultVariant(variants, 512 * gb)?.label)
    }

    @Test
    fun theDefaultIsTheSmallestWhenNothingFitsOrMemoryIsUnknown() {
        val variants = variantsOf(files)
        assertEquals("Q4_K_M", defaultVariant(variants, 2 * gb)?.label)
        assertEquals("Q4_K_M", defaultVariant(variants, null)?.label)
        assertFalse(fits(variants.first(), 2 * gb))
        assertFalse(fits(variants.first(), null))
        assertTrue(fits(variants.first(), 8 * gb))
    }

    @Test
    fun fullPrecisionIsTheDefaultOnlyWhenThereIsNothingElse() {
        assertEquals("F16", defaultVariant(variantsOf(listOf(file("model-f16.gguf", gb))), 64 * gb)?.label)
        assertNull(defaultVariant(emptyList(), 64 * gb))
    }
}
