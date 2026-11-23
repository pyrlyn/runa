package runa.core

/** Diagnostic verbosity, least to most severe. A message is shown when its level is at or above the threshold. */
public enum class LogLevel(
    public val id: String,
) {
    DEBUG("debug"),
    INFO("info"),
    WARN("warn"),
    ERROR("error"),
    ;

    public companion object {
        public fun parse(id: String): LogLevel? = entries.firstOrNull { it.id == id }
    }
}
