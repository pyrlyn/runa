package runa.hub

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class HubNameTest {
    private val repo = HubRepo("bartowski", "Qwen2.5-0.5B-Instruct-GGUF")

    @Test
    fun parsesOwnerAndRepository() {
        assertEquals(HubRequest(repo), HubRequest.parse("bartowski/Qwen2.5-0.5B-Instruct-GGUF"))
    }

    @Test
    fun quantizationAfterAtSignIsUpperCased() {
        assertEquals(HubRequest(repo, quantization = "Q4_K_M"), HubRequest.parse("bartowski/Qwen2.5-0.5B-Instruct-GGUF@q4_k_m"))
    }

    @Test
    fun parsesHubUrls() {
        val base = "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF"
        assertEquals(HubRequest(repo), HubRequest.parse(base))
        assertEquals(HubRequest(repo), HubRequest.parse("$base/"))
        assertEquals(HubRequest(repo), HubRequest.parse("https://hf.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF?library=gguf"))
        assertEquals(HubRequest(repo.copy(revision = "v2")), HubRequest.parse("$base/tree/v2"))
        assertEquals(
            HubRequest(repo, file = "sub/model-Q4_K_M.gguf"),
            HubRequest.parse("$base/blob/main/sub/model-Q4_K_M.gguf"),
        )
        assertEquals(
            HubRequest(repo, file = "model-Q4_K_M.gguf"),
            HubRequest.parse("$base/resolve/main/model-Q4_K_M.gguf?download=true"),
        )
    }

    @Test
    fun rejectsEverythingElse() {
        listOf(
            "",
            "qwen",
            "a/b/c",
            "../etc/passwd",
            "owner/..",
            "https://example.com/owner/repo",
            "https://huggingface.co/owner",
            "https://huggingface.co/owner/repo/discussions/1",
            "https://huggingface.co/../repo",
        ).forEach { assertNull(HubRequest.parse(it), it) }
    }
}
