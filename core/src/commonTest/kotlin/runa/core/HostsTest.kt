package runa.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class HostsTest {
    @Test
    fun loopbackIsThisMachineOnly() {
        for (host in listOf("127.0.0.1", "127.1.2.3", "localhost", "LOCALHOST", "::1")) assertTrue(isLoopbackHost(host), host)
        for (host in listOf("0.0.0.0", "::", "192.0.2.5", "127.example.com", "localhost.example.com", "")) {
            assertFalse(isLoopbackHost(host), host)
        }
    }

    @Test
    fun anIpv6AddressIsBracketedInAUrl() {
        assertEquals("http://127.0.0.1:7862", httpUrl("127.0.0.1", 7862))
        assertEquals("http://[::1]:7862", httpUrl("::1", 7862))
    }
}
