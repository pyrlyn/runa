package runa.config

import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import okio.FileSystem
import okio.IOException
import okio.Path
import runa.core.UserInputError
import runa.platform.writeAtomically

/**
 * `config.json` as a JSON tree. Kept as a tree, not a data class, so keys written by a newer runa
 * survive a `config set` from an older one.
 */
public class ConfigFile private constructor(
    public val path: Path,
    private val root: JsonObject,
) {
    /** The stored value in canonical string form, or null when the file does not set [key]. */
    public operator fun get(key: ConfigKey): String? {
        var node: JsonElement = root
        for (segment in key.id.split('.')) {
            node = (node as? JsonObject)?.get(segment) ?: return null
        }
        val value = (node as? JsonPrimitive)?.takeIf { it !is kotlinx.serialization.json.JsonNull }?.content
        if (value == null) throw UserInputError("$path: ${key.id} must be a ${key.kind.name.lowercase()}")
        key.problem(value)?.let { throw UserInputError("$path: ${key.id} $it") }
        return value
    }

    /** A copy with [key] set to [value]; throws [UserInputError] when the value is invalid. */
    public fun with(
        key: ConfigKey,
        value: String,
    ): ConfigFile {
        key.problem(value)?.let { throw UserInputError("${key.id} $it") }
        val primitive =
            when (key.kind) {
                ConfigKind.STRING -> JsonPrimitive(value)
                ConfigKind.INT -> JsonPrimitive(value.toInt())
                ConfigKind.BOOL -> JsonPrimitive(value.toBoolean())
            }
        return ConfigFile(path, root.withValue(key.id.split('.'), primitive))
    }

    public fun save(fileSystem: FileSystem) {
        try {
            fileSystem.writeAtomically(path) { writeUtf8(format.encodeToString(JsonObject.serializer(), root) + "\n") }
        } catch (e: IOException) {
            throw UserInputError("Cannot write $path: ${e.message}")
        }
    }

    public companion object {
        private val format = Json { prettyPrint = true }

        /** A missing file is an empty config; an unreadable or malformed one is a [UserInputError]. */
        public fun load(
            fileSystem: FileSystem,
            path: Path,
        ): ConfigFile {
            val root =
                try {
                    if (!fileSystem.exists(path)) return ConfigFile(path, JsonObject(emptyMap()))
                    format.parseToJsonElement(fileSystem.read(path) { readUtf8() }) as? JsonObject
                } catch (e: IOException) {
                    throw UserInputError("Cannot read $path: ${e.message}")
                } catch (e: SerializationException) {
                    throw UserInputError("$path is not valid JSON: ${e.message?.lineSequence()?.firstOrNull()}")
                }
            return ConfigFile(path, root ?: throw UserInputError("$path must contain a JSON object"))
        }
    }
}

private fun JsonObject.withValue(
    segments: List<String>,
    value: JsonPrimitive,
): JsonObject {
    val head = segments.first()
    val child =
        if (segments.size == 1) {
            value
        } else {
            ((this[head] as? JsonObject) ?: JsonObject(emptyMap())).withValue(segments.drop(1), value)
        }
    return JsonObject(this + (head to child))
}
