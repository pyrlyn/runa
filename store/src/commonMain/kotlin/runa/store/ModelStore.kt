package runa.store

import okio.FileMetadata
import okio.FileSystem
import okio.IOException
import okio.Path
import runa.core.ModelKey
import runa.core.ModelKind
import runa.core.UserInputError
import runa.gguf.GGUF_EXTENSION
import runa.gguf.GgufFormatException
import runa.gguf.GgufHeader
import runa.gguf.GgufSplitPart
import runa.gguf.ggufStem
import runa.gguf.isGgufFileName
import runa.gguf.quantizationInName
import runa.gguf.readGgufHeader
import runa.gguf.withoutQuantization
import runa.platform.createHardLink

/**
 * The models on disk: `<models dir>/<publisher>/<repo>/<file>.gguf`, the layout LM Studio uses.
 * Nothing here needs the daemon.
 */
public class ModelStore(
    private val fileSystem: FileSystem,
    public val modelsDir: Path,
) {
    /**
     * Every model in the store. Files that cannot be read are reported in
     * [StoreListing.problems] instead of failing the listing: one bad download must not hide the
     * rest.
     */
    public fun list(): StoreListing {
        val indexPath = modelsDir / ModelIndex.FILE_NAME
        val cached = ModelIndex.load(fileSystem, indexPath)
        val entries = LinkedHashMap<String, IndexEntry>()
        val models = mutableListOf<StoredModel>()
        val problems = mutableListOf<StoreProblem>()

        for (publisher in directoriesIn(modelsDir)) {
            for (repo in directoriesIn(publisher)) {
                val files =
                    children(repo).filter { isGgufFileName(it.name) }.mapNotNull { file ->
                        val relative = file.relativeTo(modelsDir).toString()
                        try {
                            entry(file, cached.files[relative]).also { entries[relative] = it }.let { file to it }
                        } catch (e: GgufFormatException) {
                            problems += StoreProblem(file, e.message.orEmpty())
                            null
                        } catch (e: IOException) {
                            problems += StoreProblem(file, e.message ?: "cannot be read")
                            null
                        }
                    }
                models += variants(ModelKey(publisher.name, repo.name), files, problems)
            }
        }
        if (entries != cached.files) ModelIndex(files = entries).save(fileSystem, indexPath)
        return StoreListing(models.sortedBy { it.key.toString().lowercase() }, problems)
    }

    /**
     * Checks that [source] is a GGUF model and finds the files that belong to it (all parts when it
     * is one part of a split model).
     *
     * @throws UserInputError when the file is missing, is not a GGUF or lacks a part.
     */
    public fun inspect(source: Path): ImportSource {
        val file =
            try {
                fileSystem.canonicalize(source)
            } catch (e: IOException) {
                throw UserInputError("$source does not exist")
            }
        if (fileSystem.metadataOrNull(file)?.isRegularFile != true) throw UserInputError("$source is not a file")
        if (!isGgufFileName(file.name)) throw UserInputError("$source is not a GGUF model: its name must end with $GGUF_EXTENSION")
        val header =
            try {
                readGgufHeader(fileSystem, file)
            } catch (e: GgufFormatException) {
                throw UserInputError("$source is not a GGUF model: ${e.message}")
            } catch (e: IOException) {
                throw UserInputError("Cannot read $source: ${e.message}")
            }
        val split = GgufSplitPart.parse(file.name)
        val files =
            if (split == null) {
                listOf(file)
            } else {
                val directory = checkNotNull(file.parent)
                (1..split.total).map { directory / split.fileName(it) }.onEach {
                    if (!fileSystem.exists(it)) throw UserInputError("$source is one part of a split model and ${it.name} is missing")
                }
            }
        val publisher = header.organization?.let(ModelKey::toValidName) ?: DEFAULT_PUBLISHER
        val repo = ModelKey.toValidName(withoutQuantization(ggufStem(file.name))) ?: DEFAULT_REPO
        return ImportSource(files, ModelKey(publisher, repo))
    }

    /**
     * Decides where the files of [source] go for the model [key]. Changes nothing on disk.
     *
     * @throws UserInputError when a destination already exists.
     */
    public fun planImport(
        source: ImportSource,
        key: ModelKey,
    ): ImportPlan {
        require(
            key.variant == null && ModelKey.isValidName(key.publisher) && ModelKey.isValidName(key.repo),
        ) { "Not a publisher/repo key: $key" }
        val directory = modelsDir / key.publisher / key.repo
        val steps = source.files.map { ImportPlan.Step(it, directory / it.name) }
        steps.firstOrNull { fileSystem.metadataOrNull(it.destination) != null }?.let {
            throw UserInputError("${it.destination} already exists")
        }
        return ImportPlan(key, steps)
    }

    /** Carries out [plan]. A copy becomes visible only when it is complete. */
    public fun import(
        plan: ImportPlan,
        mode: ImportMode,
    ) {
        try {
            for ((source, destination) in plan.steps) {
                fileSystem.createDirectories(checkNotNull(destination.parent))
                when (mode) {
                    ImportMode.MOVE -> move(source, destination)
                    ImportMode.COPY -> copy(source, destination)
                    ImportMode.HARD_LINK -> createHardLink(target = source, link = destination)
                    ImportMode.SYMLINK -> fileSystem.createSymlink(destination, source)
                }
            }
        } catch (e: IOException) {
            throw UserInputError("Import failed: ${e.message}")
        }
    }

    private fun move(
        source: Path,
        destination: Path,
    ) {
        try {
            fileSystem.atomicMove(source, destination)
        } catch (_: IOException) {
            // A rename cannot cross volumes; a copy can.
            copy(source, destination)
            fileSystem.delete(source)
        }
    }

    private fun copy(
        source: Path,
        destination: Path,
    ) {
        // The scan ignores this name, so a half-written copy is never listed as a model.
        val partial = checkNotNull(destination.parent) / ".${destination.name}.part"
        try {
            fileSystem.copy(source, partial)
            fileSystem.atomicMove(partial, destination)
        } catch (e: IOException) {
            fileSystem.delete(partial, mustExist = false)
            throw e
        }
    }

    private fun entry(
        file: Path,
        cached: IndexEntry?,
    ): IndexEntry {
        val metadata = resolved(file) ?: throw IOException("broken link")
        val size = metadata.size ?: 0
        val mtime = metadata.lastModifiedAtMillis ?: 0
        if (cached != null && cached.size == size && cached.mtime == mtime) return cached
        return IndexEntry(size, mtime, readGgufHeader(fileSystem, file))
    }

    private fun variants(
        model: ModelKey,
        files: List<Pair<Path, IndexEntry>>,
        problems: MutableList<StoreProblem>,
    ): List<StoredModel> {
        val groups =
            files
                .filter { (_, entry) -> entry.type == null || entry.type == GgufHeader.TYPE_MODEL }
                .groupBy { (file, _) -> GgufSplitPart.parse(file.name)?.let { "${it.stem}/${it.total}" } ?: file.name }
                .values
                .mapNotNull { parts ->
                    val total = GgufSplitPart.parse(parts.first().first.name)?.total ?: 1
                    if (parts.size == total) return@mapNotNull parts
                    problems += StoreProblem(parts.first().first, "split model with ${parts.size} of $total parts")
                    null
                }
        // The quantization is the natural variant name; file stems take over when two files share one.
        val names = groups.map { parts -> quantization(parts.first())?.lowercase() }
        return groups.mapIndexed { index, parts ->
            val (file, entry) = parts.first()
            val stem = ggufStem(file.name)
            val unique = names[index]?.takeIf { name -> names.count { it == name } == 1 }
            StoredModel(
                key = model.copy(variant = unique ?: stem.lowercase()),
                files = parts.map { it.first },
                sizeBytes = parts.sumOf { it.second.size },
                kind = if (entry.embedding) ModelKind.EMBEDDING else ModelKind.LLM,
                architecture = entry.architecture,
                name = entry.name,
                parameterCount = parts.sumOf { it.second.parameters },
                quantization = quantization(parts.first()),
                contextLength = entry.contextLength,
            )
        }
    }

    private fun quantization(file: Pair<Path, IndexEntry>): String? =
        file.second.quantization ?: quantizationInName(ggufStem(file.first.name))

    private fun directoriesIn(directory: Path): List<Path> =
        children(directory).filter { !it.name.startsWith(".") && resolved(it)?.isDirectory == true }

    private fun children(directory: Path): List<Path> = fileSystem.listOrNull(directory).orEmpty().sorted()

    /** The metadata of what [path] points to, following a symbolic link; null when nothing is there. */
    private fun resolved(path: Path): FileMetadata? =
        try {
            val metadata = fileSystem.metadataOrNull(path)
            if (metadata?.symlinkTarget == null) metadata else fileSystem.metadataOrNull(fileSystem.canonicalize(path))
        } catch (_: IOException) {
            null
        }

    private companion object {
        const val DEFAULT_PUBLISHER = "local"
        const val DEFAULT_REPO = "model"
    }
}

/** How `runa import` puts a file into the store. */
public enum class ImportMode {
    MOVE,
    COPY,
    HARD_LINK,
    SYMLINK,
}

/** A file accepted for import, with the key its name and metadata suggest. */
public class ImportSource internal constructor(
    public val files: List<Path>,
    public val suggestedKey: ModelKey,
)

public data class ImportPlan(
    val key: ModelKey,
    val steps: List<Step>,
) {
    public data class Step(
        val source: Path,
        val destination: Path,
    )
}
