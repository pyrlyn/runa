package runa.engine.llama

import runa.core.GpuOffload
import runa.gguf.GgufHeader
import kotlin.math.roundToInt

/**
 * The bytes of KV cache llama.cpp allocates for [contextLength] tokens: per layer, a key and a
 * value vector for every KV head, stored as 16-bit floats, with the context rounded up as
 * llama.cpp does. Zero when the header lacks the dimensions (not a transformer, or not a model).
 */
internal fun kvCacheBytes(
    header: GgufHeader,
    contextLength: Long,
): Long {
    val layers = header.architectureLong("block_count") ?: return 0
    val embedding = header.embeddingLength ?: return 0
    // Per-layer head counts are arrays; such a model is sized as if it had no grouped attention.
    val heads = header.architectureLong("attention.head_count")?.takeIf { it > 0 }
    val kvHeads = header.architectureLong("attention.head_count_kv")?.takeIf { it > 0 } ?: heads ?: 1
    val headLength = if (heads == null) embedding else embedding / heads
    val keyLength = header.architectureLong("attention.key_length") ?: headLength
    val valueLength = header.architectureLong("attention.value_length") ?: headLength
    val cells = (contextLength + CONTEXT_STEP - 1) / CONTEXT_STEP * CONTEXT_STEP
    return layers * cells * kvHeads * (keyLength + valueLength) * F16_BYTES
}

/**
 * `n_gpu_layers` for [gpu]. llama.cpp counts the output layer as one more layer after the
 * [blockCount] transformer blocks, and takes a negative value as "all of them".
 */
internal fun gpuLayerCount(
    gpu: GpuOffload,
    blockCount: Long?,
): Int =
    when (gpu) {
        GpuOffload.Off -> 0
        GpuOffload.Max -> ALL_LAYERS
        is GpuOffload.Ratio -> if (blockCount == null || gpu.value >= 1.0) ALL_LAYERS else (gpu.value * (blockCount + 1)).roundToInt()
    }

internal const val ALL_LAYERS: Int = -1
private const val F16_BYTES = 2L

// `GGML_PAD(cparams.n_ctx, 256)` in llama.cpp's llama-context.cpp.
private const val CONTEXT_STEP = 256L
