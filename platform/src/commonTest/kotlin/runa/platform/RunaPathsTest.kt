package runa.platform

import okio.Path.Companion.toPath
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RunaPathsTest {
    private fun paths(
        os: HostOs,
        vararg env: Pair<String, String>,
    ) = resolveRunaPaths(os, mapOf(*env)::get)

    @Test
    fun defaultsToDotRunaInHome() {
        val home = "/home/ivan/.runa".toPath()
        for (os in listOf(HostOs.MACOS, HostOs.LINUX)) {
            assertEquals(RunaPaths(home, home, home), paths(os, "HOME" to "/home/ivan"))
        }
    }

    @Test
    fun runaHomeOverridesEverything() {
        val home = "/data/runa".toPath()
        val actual =
            paths(HostOs.LINUX, "RUNA_HOME" to "/data/runa", "HOME" to "/home/ivan", "XDG_CONFIG_HOME" to "/xdg/config")
        assertEquals(RunaPaths(home, home, home), actual)
    }

    @Test
    fun linuxHonoursEachXdgVariable() {
        val actual =
            paths(
                HostOs.LINUX,
                "HOME" to "/home/ivan",
                "XDG_CONFIG_HOME" to "/xdg/config",
                "XDG_DATA_HOME" to "/xdg/data",
            )
        assertEquals("/xdg/config/runa".toPath(), actual.configDir)
        assertEquals("/xdg/data/runa".toPath(), actual.dataDir)
        assertEquals("/home/ivan/.runa".toPath(), actual.stateDir)
        assertEquals("/xdg/config/runa/config.json".toPath(), actual.configFile)
        assertEquals("/xdg/data/runa/models".toPath(), actual.modelsDir)
    }

    @Test
    fun xdgIsIgnoredOutsideLinuxAndWhenRelative() {
        val home = "/Users/ivan/.runa".toPath()
        assertEquals(home, paths(HostOs.MACOS, "HOME" to "/Users/ivan", "XDG_CONFIG_HOME" to "/xdg").configDir)
        assertEquals(home, paths(HostOs.LINUX, "HOME" to "/Users/ivan", "XDG_CONFIG_HOME" to "relative").configDir)
    }

    @Test
    fun windowsUsesLocalAppData() {
        val actual = paths(HostOs.WINDOWS, "LOCALAPPDATA" to "/c/Users/ivan/AppData/Local", "USERPROFILE" to "/c/Users/ivan")
        assertEquals("/c/Users/ivan/AppData/Local/runa".toPath(), actual.configDir)
        assertEquals("/c/Users/ivan/.runa".toPath(), paths(HostOs.WINDOWS, "USERPROFILE" to "/c/Users/ivan").dataDir)
    }

    @Test
    fun missingHomeIsAnError() {
        assertFailsWith<IllegalStateException> { paths(HostOs.MACOS) }
    }
}
