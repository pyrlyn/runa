package runa.cli

import okio.FileSystem
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ImportCommandTest {
    private val fileSystem = FileSystem.SYSTEM
    private val cli = TestCli(TestCli.temporaryHome())
    private val models = cli.home / "models"
    private val source = writeModel(cli.home / "downloads" / "Tiny-Llama-Q4_K_M.gguf")

    @AfterTest
    fun cleanUp() = fileSystem.deleteRecursively(cli.home)

    @Test
    fun movesTheFileUnderANameDerivedFromIt() {
        val destination = models / "local" / "Tiny-Llama" / "Tiny-Llama-Q4_K_M.gguf"
        val run = cli.run("import", source.toString())
        assertEquals(0, run.exitCode)
        assertEquals("$destination\n", run.stdout)
        assertEquals("Imported local/Tiny-Llama (1 file moved)\n", run.stderr)
        assertFalse(fileSystem.exists(source))
        assertContains(cli.run("ls", "--variants").stdout, "local/Tiny-Llama@q4_k_m")
    }

    @Test
    fun theOtherModesKeepTheOriginal() {
        for (mode in listOf("copy", "hard-link", "symlink")) {
            val run = cli.run("import", source.toString(), "--$mode", "--user-repo", "acme/$mode", "--json")
            val destination = models / "acme" / mode / source.name
            assertEquals(
                "{\"model\":\"acme/$mode\",\"mode\":\"$mode\",\"dry_run\":false," +
                    "\"files\":[{\"source\":\"$source\",\"destination\":\"$destination\"}]}\n",
                run.stdout,
            )
            assertTrue(fileSystem.exists(destination), mode)
            assertTrue(fileSystem.exists(source), mode)
        }
        assertEquals(source, fileSystem.metadata(models / "acme" / "symlink" / source.name).symlinkTarget)
    }

    @Test
    fun dryRunChangesNothing() {
        val run = cli.run("import", source.toString(), "--copy", "--dry-run")
        assertEquals(0, run.exitCode)
        assertEquals("${models / "local" / "Tiny-Llama" / source.name}\n", run.stdout)
        assertEquals("Dry run: 1 file would be copied as local/Tiny-Llama\n", run.stderr)
        assertFalse(fileSystem.exists(models))
        assertTrue(fileSystem.exists(source))
    }

    @Test
    fun asksForTheNameOnlyWhenSomeoneCanAnswer() {
        val asked = cli.run("import", source.toString(), "--copy", input = listOf("acme/asked"))
        assertEquals(0, asked.exitCode)
        assertContains(asked.stdout, "Import as (local/Tiny-Llama):")
        assertTrue(fileSystem.exists(models / "acme" / "asked" / source.name))

        val yes = cli.run("-y", "import", source.toString(), "--copy", input = listOf("acme/ignored"))
        assertEquals(0, yes.exitCode)
        assertTrue(fileSystem.exists(models / "local" / "Tiny-Llama" / source.name))
        assertFalse(fileSystem.exists(models / "acme" / "ignored"))
    }

    @Test
    fun namesThatCouldEscapeTheStoreAreRejected() {
        for (name in listOf("../../etc", "acme", "acme/tiny@q4", "acme/.hidden", "/abs/path")) {
            val run = cli.run("import", source.toString(), "--user-repo", name)
            assertEquals(2, run.exitCode, name)
            assertContains(run.stderr, "is not a model name")
        }
        val typed = cli.run("import", source.toString(), input = listOf("../up"))
        assertEquals(2, typed.exitCode)
        assertTrue(fileSystem.exists(source))
        assertFalse(fileSystem.exists(models))
    }

    @Test
    fun filesThatAreNotGgufModelsAreRefused() {
        val archive = cli.home / "downloads" / "archive.gguf"
        fileSystem.write(archive) { writeUtf8("PK not a model") }
        val notGguf = cli.run("import", archive.toString())
        assertEquals(2, notGguf.exitCode)
        assertEquals("Error: $archive is not a GGUF model: Not a GGUF file\n", notGguf.stderr)

        val missing = cli.run("import", (cli.home / "nope.gguf").toString())
        assertEquals(2, missing.exitCode)
        assertContains(missing.stderr, "does not exist")
        assertFalse(fileSystem.exists(models))
    }

    @Test
    fun anExistingModelFileIsNotOverwritten() {
        assertEquals(0, cli.run("import", source.toString(), "--copy").exitCode)
        val again = cli.run("import", source.toString())
        assertEquals(2, again.exitCode)
        assertContains(again.stderr, "already exists")
        assertTrue(fileSystem.exists(source))
    }
}
