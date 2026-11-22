package runa.testing

import okio.FileSystem
import okio.Path.Companion.toPath
import runa.platform.systemEnv
import kotlin.test.assertEquals

/**
 * Compares [actual] with `src/commonTest/fixtures/<name>` of the module under test (Gradle runs
 * native tests from the module directory). `RUNA_UPDATE_GOLDEN=1 ./gradlew allTests` rewrites
 * the files instead.
 */
public fun assertGolden(
    name: String,
    actual: String,
) {
    val file = "src/commonTest/fixtures".toPath() / name
    if (systemEnv("RUNA_UPDATE_GOLDEN") == "1") {
        FileSystem.SYSTEM.write(file) { writeUtf8(actual) }
        return
    }
    val expected = FileSystem.SYSTEM.read(file) { readUtf8() }
    assertEquals(expected, actual, "Output differs from $file; rerun with RUNA_UPDATE_GOLDEN=1 if the change is intended")
}
