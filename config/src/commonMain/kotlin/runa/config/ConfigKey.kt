package runa.config

import runa.core.Backend
import runa.core.LogLevel

/** How a setting is stored in `config.json`. */
public enum class ConfigKind { STRING, INT, BOOL }

/**
 * Every setting of `config.json` (PLAN.md, section 7). Values travel as strings in their canonical
 * form; [kind] decides the JSON type on disk. [check] returns what is wrong with a value, or null.
 */
public enum class ConfigKey(
    public val id: String,
    public val kind: ConfigKind,
    /** The environment variable that overrides the file, if the setting has one. */
    public val env: String? = null,
    /** Null means unset by default; [MODELS_DIR] gets its default from the resolved paths. */
    public val default: String? = null,
    private val check: (String) -> String?,
) {
    MODELS_DIR("models_dir", ConfigKind.STRING, check = ::nonBlank),
    HOST("host", ConfigKind.STRING, env = "RUNA_HOST", default = "127.0.0.1", check = ::hostName),

    /** Unset means: read the daemon's port from its discovery file. */
    PORT("port", ConfigKind.INT, env = "RUNA_PORT", check = ::tcpPort),
    BACKEND("backend", ConfigKind.STRING, env = "RUNA_BACKEND", default = Backend.RUNA.id, check = oneOf(Backend.entries.map { it.id })),
    LOG_LEVEL(
        "log_level",
        ConfigKind.STRING,
        env = "RUNA_LOG_LEVEL",
        default = LogLevel.INFO.id,
        check = oneOf(LogLevel.entries.map { it.id }),
    ),

    /** 7862 spells RUNA on a phone keypad; LM Studio's 1234 stays free for LM Studio. */
    SERVER_PORT("server.port", ConfigKind.INT, default = "7862", check = ::tcpPort),
    SERVER_BIND("server.bind", ConfigKind.STRING, default = "127.0.0.1", check = ::hostName),
    SERVER_CORS("server.cors", ConfigKind.BOOL, default = "false", check = oneOf(listOf("true", "false"))),

    /** Unset means: the model's own context length. */
    CONTEXT_LENGTH("context_length", ConfigKind.INT, check = { intAtLeast(it, 1) }),
    GPU("gpu", ConfigKind.STRING, default = "max", check = ::gpuOffload),

    /** Seconds of idleness before a model is unloaded; unset means never. */
    TTL("ttl", ConfigKind.INT, check = { intAtLeast(it, 0) }),
    ;

    /** The global flag that overrides this setting, as shown in error messages. */
    public val flag: String get() = "--" + id.replace('_', '-')

    /** Returns null when [value] is valid, otherwise what is wrong with it. */
    public fun problem(value: String): String? = check(value)

    public companion object {
        public fun parse(id: String): ConfigKey? = entries.firstOrNull { it.id == id }
    }
}

private fun nonBlank(value: String): String? = if (value.isBlank()) "must not be empty" else null

private fun hostName(value: String): String? =
    if (value.isBlank() || value.any { it.isWhitespace() }) "must be a host name or an IP address" else null

private fun tcpPort(value: String): String? = if (value.toIntOrNull() in 1..65535) null else "must be a port number from 1 to 65535"

private fun intAtLeast(
    value: String,
    min: Int,
): String? = if ((value.toIntOrNull() ?: Int.MIN_VALUE) >= min) null else "must be a whole number, at least $min"

private fun oneOf(choices: List<String>): (String) -> String? =
    {
        if (it in
            choices
        ) {
            null
        } else {
            "must be one of: ${choices.joinToString(", ")}"
        }
    }

private fun gpuOffload(value: String): String? {
    val ratio = value.toDoubleOrNull()
    return if (value == "off" || value == "max" ||
        (ratio != null && ratio in 0.0..1.0)
    ) {
        null
    } else {
        "must be off, max, or a number from 0 to 1"
    }
}
