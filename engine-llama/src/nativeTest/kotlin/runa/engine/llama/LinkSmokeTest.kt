package runa.engine.llama

import kotlinx.cinterop.toKString
import runa.engine.llama.cinterop.llama_backend_free
import runa.engine.llama.cinterop.llama_backend_init
import runa.engine.llama.cinterop.llama_print_system_info
import kotlin.test.Test
import kotlin.test.assertTrue

class LinkSmokeTest {
    @Test
    fun llamaIsLinkedIn() {
        llama_backend_init()
        val info = llama_print_system_info()?.toKString().orEmpty()
        llama_backend_free()
        println(info)
        assertTrue(info.isNotEmpty())
    }
}
