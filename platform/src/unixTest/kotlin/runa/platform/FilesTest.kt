package runa.platform

import okio.FileSystem
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class FilesTest {
    private val fileSystem = FileSystem.SYSTEM
    private val directory = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-files-${Random.nextLong().toULong()}"

    @AfterTest
    fun cleanUp() = fileSystem.deleteRecursively(directory)

    @Test
    fun aLinkHasTheMetadataOfItsTarget() {
        fileSystem.createDirectories(directory)
        val file = directory / "file"
        fileSystem.write(file) { writeUtf8("x".repeat(1000)) }
        val link = directory / "link"
        fileSystem.createSymlink(link, file)

        assertEquals(1000, fileSystem.resolvedMetadataOrNull(file)?.size)
        assertEquals(1000, fileSystem.resolvedMetadataOrNull(link)?.size)

        fileSystem.delete(file)
        assertNull(fileSystem.resolvedMetadataOrNull(link))
        assertNull(fileSystem.resolvedMetadataOrNull(file))
    }
}
