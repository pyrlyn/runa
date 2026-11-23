package runa.store

import okio.FileSystem
import okio.ForwardingFileSystem
import okio.Path
import okio.Source
import runa.core.ModelKey
import runa.core.ModelKind
import runa.core.UserInputError
import runa.gguf.GgufWriter
import runa.gguf.isGgufFileName
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ModelStoreTest {
    /** Counts how many `.gguf` files are opened, to see what the index saves. */
    private class CountingFileSystem : ForwardingFileSystem(SYSTEM) {
        var modelsOpened = 0

        override fun source(file: Path): Source {
            if (isGgufFileName(file.name)) modelsOpened++
            return super.source(file)
        }
    }

    private val fileSystem = CountingFileSystem()

    // Canonical, because the store resolves links and macOS keeps its temporary directory behind one.
    private val root: Path =
        (FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-store-test-${Random.nextLong().toULong()}").let {
            fileSystem.createDirectories(it)
            fileSystem.canonicalize(it)
        }
    private val modelsDir = root / "models"
    private val store = ModelStore(fileSystem, modelsDir)

    @AfterTest
    fun cleanUp() = fileSystem.deleteRecursively(root)

    private fun llm(fileType: Long? = 15) =
        GgufWriter()
            .put("general.architecture", "llama")
            .put("general.name", "Tiny")
            .put("llama.context_length", 2048)
            .tensor("weights", 10, 100)
            .apply { if (fileType != null) put("general.file_type", fileType) }

    private fun write(
        path: Path,
        writer: GgufWriter = llm(),
    ): Path {
        fileSystem.createDirectories(checkNotNull(path.parent))
        fileSystem.write(path) { writer.writeTo(this) }
        return path
    }

    private fun keys() = store.list().models.map { it.key.toString() }

    @Test
    fun anEmptyOrMissingModelsDirListsNothing() {
        assertEquals(StoreListing(emptyList(), emptyList()), store.list())
        assertFalse(fileSystem.exists(modelsDir), "listing must not create the models dir")
    }

    @Test
    fun listsModelsWithTheirMetadata() {
        val q4 = write(modelsDir / "acme" / "tiny" / "tiny-Q4_K_M.gguf")
        write(modelsDir / "acme" / "tiny" / "tiny-Q8_0.gguf", llm(fileType = 7))
        write(modelsDir / "Nomic" / "embed" / "embed.gguf", GgufWriter().put("general.architecture", "nomic-bert"))

        val listing = store.list()
        assertEquals(emptyList(), listing.problems)
        assertEquals(listOf("acme/tiny@q4_k_m", "acme/tiny@q8_0", "Nomic/embed@embed"), listing.models.map { it.key.toString() })
        assertEquals(
            StoredModel(
                key = ModelKey("acme", "tiny", "q4_k_m"),
                files = listOf(q4),
                sizeBytes = fileSystem.metadata(q4).size!!,
                kind = ModelKind.LLM,
                architecture = "llama",
                name = "Tiny",
                parameterCount = 1000,
                quantization = "Q4_K_M",
                contextLength = 2048,
            ),
            listing.models.first(),
        )
        assertEquals(ModelKind.EMBEDDING, listing.models.last().kind)
        assertNull(listing.models.last().quantization)
    }

    @Test
    fun aWarmIndexAvoidsReadingModelFiles() {
        repeat(200) { write(modelsDir / "acme" / "model-$it" / "model-$it-Q4_K_M.gguf") }
        val cold = store.list()
        assertEquals(200, fileSystem.modelsOpened)

        val warm = ModelStore(fileSystem, modelsDir).list()
        assertEquals(200, fileSystem.modelsOpened, "a warm listing must not open a model file")
        assertEquals(cold, warm)

        write(modelsDir / "acme" / "model-7" / "model-7-Q4_K_M.gguf", llm().put("general.description", "changed"))
        store.list()
        assertEquals(201, fileSystem.modelsOpened, "only the changed file is read again")
    }

    @Test
    fun aBrokenIndexIsRebuilt() {
        write(modelsDir / "acme" / "tiny" / "tiny-Q4_K_M.gguf")
        store.list()
        for (content in listOf("not json", "{\"version\":999,\"files\":{}}", "[]")) {
            fileSystem.write(modelsDir / ".index.json") { writeUtf8(content) }
            assertEquals(listOf("acme/tiny@q4_k_m"), keys(), "index: $content")
        }
        assertContains(fileSystem.read(modelsDir / ".index.json") { readUtf8() }, "acme/tiny/tiny-Q4_K_M.gguf")
    }

    @Test
    fun unreadableFilesAreReportedAndTheRestIsListed() {
        write(modelsDir / "acme" / "tiny" / "tiny-Q4_K_M.gguf")
        val zip = modelsDir / "acme" / "tiny" / "archive.gguf"
        fileSystem.write(zip) { writeUtf8("PK not a model") }
        val cut = modelsDir / "acme" / "cut" / "cut.gguf"
        val whole = fileSystem.read(write(cut)) { readByteString() }
        fileSystem.write(cut) { write(whole.substring(0, 30)) }
        val dangling = modelsDir / "acme" / "gone" / "gone.gguf"
        fileSystem.createDirectories(dangling.parent!!)
        fileSystem.createSymlink(dangling, root / "nowhere.gguf")

        val listing = store.list()
        assertEquals(listOf("acme/tiny@q4_k_m"), listing.models.map { it.key.toString() })
        assertEquals(
            listOf(
                StoreProblem(cut, "The file ends inside its GGUF header"),
                StoreProblem(dangling, "broken link"),
                StoreProblem(zip, "Not a GGUF file"),
            ),
            listing.problems,
        )
    }

    @Test
    fun filesThatAreNotModelsAreIgnored() {
        val repo = modelsDir / "acme" / "vision"
        write(repo / "vision-Q4_K_M.gguf")
        write(repo / "mmproj-vision-f16.gguf", GgufWriter().put("general.type", "mmproj").put("general.architecture", "clip"))
        write(repo / ".hidden.gguf")
        fileSystem.write(repo / "README.md") { writeUtf8("notes") }
        write(modelsDir / "stray.gguf")
        write(modelsDir / ".cache" / "repo" / "cached.gguf")
        assertEquals(StoreListing(store.list().models, emptyList()), store.list())
        assertEquals(listOf("acme/vision@q4_k_m"), keys())
    }

    @Test
    fun aSplitModelIsOneVariantAndAnIncompleteOneIsAProblem() {
        val repo = modelsDir / "acme" / "big"
        val parts = (1..3).map { write(repo / "big-Q4_K_M-0000$it-of-00003.gguf") }
        val lonely = write(repo / "big-Q8_0-00001-of-00002.gguf", llm(fileType = 7))

        val listing = store.list()
        val model = listing.models.single()
        assertEquals("acme/big@q4_k_m", model.key.toString())
        assertEquals(parts, model.files)
        assertEquals(3000, model.parameterCount)
        assertEquals(parts.sumOf { fileSystem.metadata(it).size!! }, model.sizeBytes)
        assertEquals(listOf(StoreProblem(lonely, "split model with 1 of 2 parts")), listing.problems)
    }

    @Test
    fun variantNamesFallBackToTheFileName() {
        val repo = modelsDir / "acme" / "mix"
        write(repo / "story-Q4_K_M.gguf")
        write(repo / "code-Q4_K_M.gguf")
        write(repo / "old.IQ3_XXS.gguf", llm(fileType = null))
        write(repo / "plain.gguf", llm(fileType = null))
        assertEquals(listOf("acme/mix@code-q4_k_m", "acme/mix@iq3_xxs", "acme/mix@plain", "acme/mix@story-q4_k_m"), keys())
        assertEquals(
            "IQ3_XXS",
            store
                .list()
                .models
                .first { it.key.variant == "iq3_xxs" }
                .quantization,
        )
    }

    @Test
    fun inspectAcceptsOnlyGgufModels() {
        fun failure(path: Path) = assertFailsWith<UserInputError> { store.inspect(path) }.message.orEmpty()

        assertContains(failure(root / "missing.gguf"), "does not exist")
        assertContains(failure(root), "is not a file")
        assertContains(failure(write(root / "model.bin")), "its name must end with .gguf")
        val zip = root / "archive.gguf"
        fileSystem.write(zip) { writeUtf8("PK not a model") }
        assertContains(failure(zip), "is not a GGUF model: Not a GGUF file")
        assertContains(failure(write(root / "big-00001-of-00002.gguf")), "big-00002-of-00002.gguf is missing")
    }

    @Test
    fun inspectSuggestsAKeyFromTheMetadataAndTheFileName() {
        assertEquals(ModelKey("local", "Tiny-Llama"), store.inspect(write(root / "Tiny-Llama-Q4_K_M.gguf")).suggestedKey)
        val branded = write(root / "My Model (v2).Q8_0.gguf", llm().put("general.organization", "Acme Labs"))
        assertEquals(ModelKey("Acme-Labs", "My-Model-v2"), store.inspect(branded).suggestedKey)

        val first = write(root / "big-Q4_K_M-00001-of-00002.gguf")
        val second = write(root / "big-Q4_K_M-00002-of-00002.gguf")
        val split = store.inspect(second)
        assertEquals(listOf(first, second), split.files)
        assertEquals(ModelKey("local", "big"), split.suggestedKey)
    }

    private fun import(
        mode: ImportMode,
        name: String = "tiny-Q4_K_M.gguf",
    ): Pair<Path, Path> {
        val source = write(root / "downloads" / name)
        val plan = store.planImport(store.inspect(source), ModelKey("acme", mode.name.lowercase()))
        store.import(plan, mode)
        return source to plan.steps.single().destination
    }

    @Test
    fun moveRemovesTheSource() {
        val (source, destination) = import(ImportMode.MOVE)
        assertEquals(modelsDir / "acme" / "move" / "tiny-Q4_K_M.gguf", destination)
        assertFalse(fileSystem.exists(source))
        assertEquals(listOf("acme/move@q4_k_m"), keys())
    }

    @Test
    fun copyKeepsTheSourceAndLeavesNoPartialFile() {
        val (source, destination) = import(ImportMode.COPY)
        assertEquals(fileSystem.read(source) { readByteString() }, fileSystem.read(destination) { readByteString() })
        assertEquals(listOf(destination), fileSystem.list(destination.parent!!))
        assertNull(fileSystem.metadata(destination).symlinkTarget)
    }

    @Test
    fun hardLinkSharesTheContentWithTheSource() {
        val (source, destination) = import(ImportMode.HARD_LINK)
        assertNull(fileSystem.metadata(destination).symlinkTarget)
        // A write through one name must be visible through the other: that is what a hard link is.
        fileSystem.write(source) { writeUtf8("changed") }
        assertEquals("changed", fileSystem.read(destination) { readUtf8() })
    }

    @Test
    fun symlinkPointsAtTheSourceAndIsListed() {
        val (source, destination) = import(ImportMode.SYMLINK)
        assertEquals(source, fileSystem.metadata(destination).symlinkTarget)
        val model = store.list().models.single()
        assertEquals("acme/symlink@q4_k_m", model.key.toString())
        assertEquals(fileSystem.metadata(source).size, model.sizeBytes)
    }

    @Test
    fun aSplitModelIsImportedWhole() {
        val parts = (1..2).map { write(root / "downloads" / "big-Q4_K_M-0000$it-of-00002.gguf") }
        val plan = store.planImport(store.inspect(parts.first()), ModelKey("acme", "big"))
        assertEquals(parts.map { modelsDir / "acme" / "big" / it.name }, plan.steps.map { it.destination })
        store.import(plan, ImportMode.COPY)
        assertEquals(
            2,
            store
                .list()
                .models
                .single()
                .files.size,
        )
    }

    @Test
    fun anExistingDestinationIsNeverOverwritten() {
        import(ImportMode.COPY)
        val again = store.inspect(root / "downloads" / "tiny-Q4_K_M.gguf")
        val error = assertFailsWith<UserInputError> { store.planImport(again, ModelKey("acme", "copy")) }
        assertContains(error.message.orEmpty(), "already exists")
        assertNotNull(store.planImport(again, ModelKey("acme", "other")))
        assertTrue(fileSystem.exists(root / "downloads" / "tiny-Q4_K_M.gguf"))
    }
}
