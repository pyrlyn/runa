package runa.cli

import okio.FileSystem
import runa.gguf.GgufWriter
import runa.testing.assertGolden
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals

class LsCommandTest {
    private val cli = TestCli(TestCli.temporaryHome())
    private val models = cli.home / "models"

    @AfterTest
    fun cleanUp() = FileSystem.SYSTEM.deleteRecursively(cli.home)

    private fun threeModels() {
        writeModel(models / "acme" / "tiny" / "tiny-Q4_K_M.gguf")
        writeModel(models / "acme" / "tiny" / "tiny-Q8_0.gguf", tinyLlama(fileType = 7).tensor("extra", 500, 1000))
        val embedder =
            GgufWriter()
                .put(
                    "general.architecture",
                    "nomic-bert",
                ).put("nomic-bert.context_length", 512)
                .tensor("w", 137, 1_000_000)
        writeModel(models / "Nomic" / "embed" / "embed-f16.gguf", embedder)
    }

    /** Output with the random test directory replaced, so it can be compared with a fixture. */
    private fun stable(vararg args: String): String {
        val run = cli.run(*args)
        assertEquals("", run.stderr)
        assertEquals(0, run.exitCode)
        return run.stdout.replace(cli.home.toString(), "<home>")
    }

    @Test
    fun anEmptyStoreSaysHowToAddAModel() {
        val run = cli.run("ls")
        assertEquals(0, run.exitCode)
        assertEquals("", run.stdout)
        assertEquals("No models in $models. Add one with: runa import <file.gguf>\n", run.stderr)
        assertEquals("[]\n", stable("ls", "--json"))
    }

    @Test
    fun listsOneRowPerModel() {
        threeModels()
        assertGolden("ls.txt", stable("ls"))
    }

    @Test
    fun variantsListsOneRowPerFile() {
        threeModels()
        assertGolden("ls-variants.txt", stable("ls", "--variants"))
    }

    @Test
    fun jsonHasOneEntryPerFileWithEveryField() {
        threeModels()
        assertGolden("ls.json", stable("ls", "--json"))
        assertEquals(stable("ls", "--json"), stable("--json", "ls"))
    }

    @Test
    fun typeFiltersNarrowTheList() {
        threeModels()
        assertEquals(listOf("MODEL", "acme/tiny"), stable("ls", "--llm").lines().filter { it.isNotEmpty() }.map { it.substringBefore(' ') })
        assertEquals(
            listOf("MODEL", "Nomic/embed"),
            stable("ls", "--embedding").lines().filter { it.isNotEmpty() }.map { it.substringBefore(' ') },
        )
        val both = cli.run("ls", "--llm", "--embedding")
        assertEquals(2, both.exitCode)
        assertContains(both.stderr, "--llm and --embedding cannot be used together")
    }

    @Test
    fun aKeyShowsTheVariantsOfOneModel() {
        threeModels()

        fun keys(key: String) =
            stable("ls", key)
                .lines()
                .drop(1)
                .filter { it.isNotEmpty() }
                .map { it.substringBefore(' ') }

        assertEquals(listOf("acme/tiny@q4_k_m", "acme/tiny@q8_0"), keys("acme/tiny"))
        assertEquals(listOf("acme/tiny@q4_k_m", "acme/tiny@q8_0"), keys("TINY"))
        assertEquals(listOf("acme/tiny@q8_0"), keys("acme/tiny@Q8_0"))

        val missing = cli.run("ls", "nope")
        assertEquals(2, missing.exitCode)
        assertEquals("Error: No model matches 'nope' in $models\n", missing.stderr)
    }

    @Test
    fun aBrokenFileIsAWarningNotAFailure() {
        threeModels()
        val broken = models / "acme" / "tiny" / "broken.gguf"
        FileSystem.SYSTEM.write(broken) { writeUtf8("not a model") }
        val run = cli.run("ls", "--json")
        assertEquals(0, run.exitCode)
        assertEquals("Warning: Skipped $broken: Not a GGUF file\n", run.stderr)
        assertEquals(3, run.stdout.split("\"key\"").size - 1)
        assertEquals("", cli.run("--quiet", "ls").stderr)
    }

    @Test
    fun theModelsDirSettingIsHonoured() {
        val elsewhere = cli.home / "elsewhere"
        writeModel(elsewhere / "acme" / "tiny" / "tiny-Q4_K_M.gguf")
        assertEquals(0, cli.run("config", "set", "models_dir", elsewhere.toString()).exitCode)
        assertContains(stable("ls"), "acme/tiny")
    }
}
