package runa.cli

import okio.FileSystem
import okio.Path
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

    /** LM Studio's folder: two GGUF models, one beside its projector, and a model that is not GGUF. */
    private fun lmStudioModels(root: Path = cli.home / "lmstudio" / "models"): Path {
        writeModel(root / "acme" / "tiny-GGUF" / "tiny-Q4_K_M.gguf")
        writeModel(root / "acme" / "tiny-GGUF" / "mmproj-tiny-f16.gguf")
        writeModel(root / "other" / "small-GGUF" / "small-Q8_0.gguf", tinyLlama(fileType = 7))
        fileSystem.createDirectories(root / "prism" / "bonsai-mlx")
        fileSystem.write(root / "prism" / "bonsai-mlx" / "model.safetensors") { writeUtf8("weights") }
        return root
    }

    @Test
    fun fromLmStudioSymlinksEveryGgufModelUnderItsName() {
        val root = lmStudioModels()
        val tiny = models / "acme" / "tiny-GGUF" / "tiny-Q4_K_M.gguf"
        val small = models / "other" / "small-GGUF" / "small-Q8_0.gguf"

        val run = cli.run("import", "--from-lmstudio=$root")

        assertEquals(0, run.exitCode)
        assertEquals("$tiny\n$small\n", run.stdout)
        assertEquals("Imported 2 models from $root (symlinked)\n", run.stderr)
        assertEquals(root / "acme" / "tiny-GGUF" / "tiny-Q4_K_M.gguf", fileSystem.metadata(tiny).symlinkTarget)
        // The projector is no model, and what is not GGUF is not runa's to run.
        assertEquals(listOf(tiny), fileSystem.list(models / "acme" / "tiny-GGUF"))
        assertFalse(fileSystem.exists(models / "prism"))
        val listed = cli.run("ls").stdout
        assertContains(listed, "acme/tiny-GGUF")
        assertContains(listed, "other/small-GGUF")
    }

    @Test
    fun fromLmStudioLeavesAloneWhatTheStoreHas() {
        val root = lmStudioModels()
        assertEquals(0, cli.run("import", "--from-lmstudio=$root").exitCode)
        writeModel(root / "acme" / "new-GGUF" / "new-Q4_K_M.gguf")

        val again = cli.run("import", "--from-lmstudio=$root", "--json")
        val file = "new-Q4_K_M.gguf"
        assertEquals(0, again.exitCode)
        assertEquals("Imported 1 model from $root (symlinked); 2 already in the store\n", again.stderr)
        assertEquals(
            "[{\"model\":\"acme/new-GGUF\",\"mode\":\"symlink\",\"dry_run\":false," +
                "\"files\":[{\"source\":\"${root / "acme" / "new-GGUF" / file}\"," +
                "\"destination\":\"${models / "acme" / "new-GGUF" / file}\"}]}]\n",
            again.stdout,
        )

        val nothingNew = cli.run("import", "--from-lmstudio=$root")
        assertEquals("", nothingNew.stdout)
        assertEquals("Every GGUF model in $root is already in the store\n", nothingNew.stderr)
    }

    @Test
    fun fromLmStudioLooksInItsDefaultFolder() {
        val user = cli.home / "user"
        lmStudioModels(user / ".lmstudio" / "models")

        val run = cli.run("import", "--from-lmstudio", "--dry-run", env = mapOf("HOME" to user.toString()))

        assertEquals(0, run.exitCode)
        assertEquals("Dry run: 2 models would be symlinked from ${user / ".lmstudio" / "models"}\n", run.stderr)
        assertFalse(fileSystem.exists(models))
    }

    @Test
    fun fromLmStudioPassesOverAFileThatIsNoModel() {
        val root = lmStudioModels()
        val broken = root / "acme" / "broken-GGUF" / "broken.gguf"
        fileSystem.createDirectories(checkNotNull(broken.parent))
        fileSystem.write(broken) { writeUtf8("PK not a model") }

        val run = cli.run("import", "--from-lmstudio=$root", "--copy")

        assertEquals(0, run.exitCode)
        assertContains(run.stderr, "Warning: Skipped: $broken is not a GGUF model: Not a GGUF file\n")
        assertContains(run.stderr, "Imported 2 models from $root (copied)\n")
        // A copy, as asked, is a file of its own.
        assertEquals(null, fileSystem.metadata(models / "acme" / "tiny-GGUF" / "tiny-Q4_K_M.gguf").symlinkTarget)
    }

    @Test
    fun fromLmStudioNeedsItsFolderAndNothingElse() {
        val missing = cli.run("import", "--from-lmstudio=${cli.home / "nowhere"}")
        assertEquals(2, missing.exitCode)
        assertEquals(
            "Error: LM Studio's models folder is not at ${cli.home / "nowhere"}; name it with --from-lmstudio=<dir>\n",
            missing.stderr,
        )

        val both = cli.run("import", source.toString(), "--from-lmstudio=${lmStudioModels()}")
        assertEquals(2, both.exitCode)
        assertContains(both.stderr, "--from-lmstudio takes every model as it is")
        assertTrue(fileSystem.exists(source))

        val neither = cli.run("import")
        assertEquals(2, neither.exitCode)
        assertContains(neither.stderr, "Name the .gguf file to import, or use --from-lmstudio")
    }
}
