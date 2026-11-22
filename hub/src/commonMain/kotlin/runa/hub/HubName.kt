package runa.hub

import runa.core.ModelKey

/** A model repository on the Hub at one revision. */
public data class HubRepo(
    val owner: String,
    val name: String,
    val revision: String = DEFAULT_REVISION,
) {
    /** `owner/name`, as the Hub writes it. */
    val id: String get() = "$owner/$name"

    public companion object {
        public const val DEFAULT_REVISION: String = "main"
    }
}

/** What `runa get <name>` asks for. */
public data class HubRequest(
    val repo: HubRepo,
    /** From `owner/repo@quant`: the wanted quantization, in upper case. */
    val quantization: String? = null,
    /** From a URL that points at one file: its path in the repository. */
    val file: String? = null,
) {
    public companion object {
        /**
         * Parses `owner/repo`, `owner/repo@quant`, or a Hub URL of a repository, a tree or a file.
         * Null when [text] is none of these. Owner and repository names become directories in the
         * model store, so they must pass [ModelKey.isValidName].
         */
        public fun parse(text: String): HubRequest? {
            val url = URL_PREFIXES.firstOrNull { text.startsWith(it, ignoreCase = true) }
            if (url == null) {
                val key = ModelKey.parse(text) ?: return null
                return HubRequest(HubRepo(key.publisher, key.repo), key.variant?.uppercase())
            }
            val segments =
                text
                    .drop(url.length)
                    .substringBefore('?')
                    .substringBefore('#')
                    .split('/')
                    .filter { it.isNotEmpty() }
            if (segments.size < 2 || !segments.take(2).all(ModelKey::isValidName)) return null
            val (owner, name) = segments
            return when {
                segments.size == 2 -> {
                    HubRequest(HubRepo(owner, name))
                }

                segments[2] == "tree" && segments.size == 4 -> {
                    HubRequest(HubRepo(owner, name, segments[3]))
                }

                segments[2] in FILE_VIEWS && segments.size > 4 -> {
                    HubRequest(HubRepo(owner, name, segments[3]), file = segments.drop(4).joinToString("/"))
                }

                else -> {
                    null
                }
            }
        }

        private val URL_PREFIXES = listOf("https://huggingface.co/", "https://hf.co/")
        private val FILE_VIEWS = setOf("blob", "resolve")
    }
}
