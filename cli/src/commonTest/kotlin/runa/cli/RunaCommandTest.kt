package runa.cli

import com.github.ajalt.clikt.testing.test
import kotlin.test.Test
import kotlin.test.assertEquals

class RunaCommandTest {
    @Test
    fun versionPrintsVersionAndCommit() {
        for (flag in listOf("--version", "-v")) {
            val result = RunaCommand(version = "1.2.3", gitSha = "abc123").test(flag)
            assertEquals("runa 1.2.3 (abc123)\n", result.stdout)
            assertEquals(0, result.statusCode)
        }
    }
}
