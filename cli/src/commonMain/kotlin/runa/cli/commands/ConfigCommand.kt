package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.convert
import com.github.ajalt.clikt.parameters.arguments.optional
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.choice
import com.github.ajalt.mordant.widgets.Text
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive
import runa.cli.Globals
import runa.cli.output.listing
import runa.config.ConfigKey
import runa.config.ConfigKind
import runa.config.ResolvedSetting

internal class ConfigCommand : CliktCommand(name = "config") {
    init {
        subcommands(Get(), Set(), PathCommand())
    }

    override fun help(context: Context): String = "Show and change settings (config.json)"

    override fun run() = Unit

    @Serializable
    private data class Entry(
        val key: String,
        val value: JsonElement,
        val source: String,
    )

    @Serializable
    private data class Location(
        val path: String,
    )

    private class Get : CliktCommand(name = "get") {
        private val globals by requireObject<Globals>()
        private val key by argument("key", help = KEY_HELP).choice(keysById).optional()
        private val json by option("--json", help = JSON_HELP).flag()

        override fun help(context: Context): String = "Print one setting, or all of them, after flags and environment are applied"

        override fun run() {
            globals.log.debug { "Reading ${globals.paths.configFile}" }
            val config = globals.config
            val renderer = globals.renderer(json)
            val one = key
            if (one != null) {
                renderer.render(config.resolve(one).toEntry(), Entry.serializer()) { Text(it.value.display()) }
            } else {
                renderer.render(ConfigKey.entries.map { config.resolve(it).toEntry() }, ListSerializer(Entry.serializer())) { entries ->
                    listing(listOf("KEY", "VALUE", "SOURCE"), entries.map { listOf(it.key, it.value.display(), it.source) })
                }
            }
        }

        private fun ResolvedSetting.toEntry(): Entry {
            val text = value
            val element =
                when {
                    text == null -> JsonNull
                    key.kind == ConfigKind.INT -> JsonPrimitive(text.toInt())
                    key.kind == ConfigKind.BOOL -> JsonPrimitive(text.toBoolean())
                    else -> JsonPrimitive(text)
                }
            return Entry(key.id, element, source.id)
        }

        private fun JsonElement.display(): String = (this as? JsonPrimitive)?.takeIf { it !is JsonNull }?.content.orEmpty()
    }

    private class Set : CliktCommand(name = "set") {
        private val globals by requireObject<Globals>()
        private val key by argument("key", help = KEY_HELP).choice(keysById)
        private val value by argument("value").convert { it.trim() }

        override fun help(context: Context): String = "Save a setting to config.json"

        override fun run() {
            val updated = globals.loadConfigFile().with(key, value)
            updated.save(globals.environment.fileSystem)
            globals.log.info { "Set ${key.id} = $value in ${updated.path}" }
        }
    }

    private class PathCommand : CliktCommand(name = "path") {
        private val globals by requireObject<Globals>()
        private val json by option("--json", help = JSON_HELP).flag()

        override fun help(context: Context): String = "Print where config.json lives"

        override fun run() {
            val location = Location(globals.paths.configFile.toString())
            globals.renderer(json).render(location, Location.serializer()) { Text(it.path) }
        }
    }

    private companion object {
        val keysById: Map<String, ConfigKey> = ConfigKey.entries.associateBy { it.id }
        const val KEY_HELP: String = "A setting name, for example server.port"
    }
}
