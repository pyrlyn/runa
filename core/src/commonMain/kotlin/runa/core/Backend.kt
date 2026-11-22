package runa.core

/** Which service a command talks to: runa's own daemon or a running LM Studio. */
public enum class Backend(
    public val id: String,
) {
    RUNA("runa"),
    LMSTUDIO("lmstudio"),
    ;

    public companion object {
        public fun parse(id: String): Backend? = entries.firstOrNull { it.id == id }
    }
}
