package runa.config

import runa.core.UserInputError
import runa.platform.RunaPaths

/** Where an effective setting came from, highest precedence first. */
public enum class ConfigSource(
    public val id: String,
) {
    FLAG("flag"),
    ENV("env"),
    FILE("file"),
    DEFAULT("default"),
}

/** An effective setting; [value] is null only for settings that are unset by default. */
public data class ResolvedSetting(
    val key: ConfigKey,
    val value: String?,
    val source: ConfigSource,
)

/** Applies the precedence rule of PLAN.md, section 7: flag > environment > `config.json` > default. */
public class ConfigResolver(
    private val flags: Map<ConfigKey, String>,
    private val env: (String) -> String?,
    /** Read on first use, so commands that need no settings work with a broken `config.json`. */
    file: () -> ConfigFile,
    private val paths: RunaPaths,
) {
    private val file by lazy(file)

    public fun resolve(key: ConfigKey): ResolvedSetting {
        flags[key]?.let { return ResolvedSetting(key, checked(key, it, key.flag), ConfigSource.FLAG) }
        key.env?.let { name ->
            env(name)?.takeIf { it.isNotEmpty() }?.let {
                return ResolvedSetting(key, checked(key, it, name), ConfigSource.ENV)
            }
        }
        file[key]?.let { return ResolvedSetting(key, it, ConfigSource.FILE) }
        val default = if (key == ConfigKey.MODELS_DIR) paths.modelsDir.toString() else key.default
        return ResolvedSetting(key, default, ConfigSource.DEFAULT)
    }

    /**
     * Rejects invalid flag and environment values without reading `config.json`. Call it before a
     * command does any work, so a typo in `--port` cannot fail halfway through.
     */
    public fun checkOverrides() {
        flags.forEach { (key, value) -> checked(key, value, key.flag) }
        for (key in ConfigKey.entries) {
            val name = key.env ?: continue
            env(name)?.takeIf { it.isNotEmpty() }?.let { checked(key, it, name) }
        }
    }

    private fun checked(
        key: ConfigKey,
        value: String,
        origin: String,
    ): String {
        key.problem(value)?.let { throw UserInputError("$origin $it") }
        return value
    }
}
