package runa.core

/** What a model produces; [id] is the word used by `--json` output and the API. */
public enum class ModelKind(
    public val id: String,
) {
    LLM("llm"),
    EMBEDDING("embedding"),
}

/**
 * How a local model is addressed: `publisher/repo`, the two directories under the models dir, and
 * optionally `@variant` for one file of the repo (usually a quantization, as in `@q4_k_m`).
 */
public data class ModelKey(
    val publisher: String,
    val repo: String,
    val variant: String? = null,
) {
    /** The key without the variant. */
    val model: String get() = "$publisher/$repo"

    override fun toString(): String = if (variant == null) model else "$model@$variant"

    public companion object {
        /** Parses what a user typed; null unless every part passes [isValidName]. */
        public fun parse(text: String): ModelKey? {
            val variant = if ('@' in text) text.substringAfter('@') else null
            val names = text.substringBefore('@').split('/')
            if (names.size != 2 || !(names + listOfNotNull(variant)).all(::isValidName)) return null
            return ModelKey(names[0], names[1], variant)
        }

        /**
         * The names runa creates on disk: letters, digits, `.`, `_` and `-`, starting with a letter
         * or digit. That excludes path separators, `..` and hidden files, so a key from user input
         * can never point outside the models dir.
         */
        public fun isValidName(name: String): Boolean = name.length <= MAX_NAME_LENGTH && NAME.matches(name) && ".." !in name

        /** Turns arbitrary text into a valid name, or null when nothing usable is left. */
        public fun toValidName(text: String): String? =
            text
                .replace(INVALID_RUN, "-")
                .replace(DOT_RUN, ".")
                .trimStart('.', '_', '-')
                .trimEnd('.', '-')
                .take(MAX_NAME_LENGTH)
                .takeIf(::isValidName)

        // Hugging Face's own limit for user and repository names.
        private const val MAX_NAME_LENGTH = 96
        private val NAME = Regex("[A-Za-z0-9][A-Za-z0-9._-]*")
        private val INVALID_RUN = Regex("[^A-Za-z0-9._-]+")
        private val DOT_RUN = Regex("\\.{2,}")
    }
}
