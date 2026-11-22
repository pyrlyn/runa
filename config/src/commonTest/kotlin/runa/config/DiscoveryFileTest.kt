package runa.config

import okio.FileSystem
import okio.Path
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class DiscoveryFileTest {
    private val directory: Path = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-discovery-test-${Random.nextLong().toULong()}"
    private val file = DiscoveryFile(FileSystem.SYSTEM, directory / "run" / "daemon.json")
    private val info = DaemonInfo(pid = 42, port = 7000, token = "secret", version = "1.2.3")

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(directory)

    @Test
    fun readsBackWhatWasWritten() {
        assertNull(file.read())
        file.write(info)
        assertEquals(info, file.read())
    }

    @Test
    fun aBrokenFileIsNoDaemon() {
        file.write(info)
        FileSystem.SYSTEM.write(file.path) { writeUtf8("{\"pid\": ") }
        assertNull(file.read())
    }

    @Test
    fun fieldsOfANewerVersionAreIgnored() {
        file.write(info)
        FileSystem.SYSTEM.write(file.path) { writeUtf8("""{"pid":42,"port":7000,"token":"secret","version":"1.2.3","started":"now"}""") }
        assertEquals(info, file.read())
    }

    @Test
    fun deletesOnlyItsOwnFile() {
        file.write(info)
        file.delete(pid = 43)
        assertEquals(info, file.read())
        file.delete(pid = 42)
        assertNull(file.read())
    }
}
