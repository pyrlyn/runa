package runa.engine.llama

import okio.Buffer
import runa.core.GpuOffload
import runa.gguf.GgufHeader
import runa.gguf.GgufWriter
import runa.gguf.readGgufHeader
import kotlin.test.Test
import kotlin.test.assertEquals

class SizingTest {
    private fun header(configure: GgufWriter.() -> Unit): GgufHeader {
        val buffer = Buffer()
        GgufWriter().put("general.architecture", "qwen2").apply(configure).writeTo(buffer)
        return readGgufHeader(buffer)
    }

    @Test
    fun kvCacheFollowsLlamaCpp() {
        // Qwen2.5-0.5B: llama.cpp v1.0.0 reports "KV self size = 48.00 MiB" for a context of 4096.
        val qwen =
            header {
                put("qwen2.block_count", 24)
                put("qwen2.embedding_length", 896)
                put("qwen2.attention.head_count", 14)
                put("qwen2.attention.head_count_kv", 2)
            }
        assertEquals(48L * 1024 * 1024, kvCacheBytes(qwen, 4096))
    }

    @Test
    fun withoutGroupedAttentionEveryHeadIsCached() {
        val model =
            header {
                put("qwen2.block_count", 2)
                put("qwen2.embedding_length", 64)
                put("qwen2.attention.head_count", 4)
            }
        assertEquals(2L * 512 * (64 + 64) * 2, kvCacheBytes(model, 512))
        // llama.cpp rounds the context up to a multiple of 256.
        assertEquals(kvCacheBytes(model, 512), kvCacheBytes(model, 257))
    }

    @Test
    fun aHeaderWithoutDimensionsHasNoCache() {
        assertEquals(0, kvCacheBytes(header {}, 4096))
    }

    @Test
    fun gpuLayersFollowTheOffloadSetting() {
        assertEquals(0, gpuLayerCount(GpuOffload.Off, 24))
        assertEquals(ALL_LAYERS, gpuLayerCount(GpuOffload.Max, 24))
        assertEquals(ALL_LAYERS, gpuLayerCount(GpuOffload.Ratio(1.0), 24))
        assertEquals(13, gpuLayerCount(GpuOffload.Ratio(0.5), 24))
        assertEquals(0, gpuLayerCount(GpuOffload.Ratio(0.0), 24))
        assertEquals(ALL_LAYERS, gpuLayerCount(GpuOffload.Ratio(0.5), null))
    }
}
