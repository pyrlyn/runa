package runa.daemon

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import runa.core.ChatMessage

/** A conversation up to and including one answer of the model. */
internal class Conversation(
    val systemPrompt: String?,
    val messages: List<ChatMessage>,
)

/**
 * The conversations a chat request can continue, each known by the id of the answer it ends with.
 * Kept in memory, and only the [capacity] used last: they are gone with the daemon, and a client
 * that names a forgotten one is told so rather than answered without its history.
 */
internal class Conversations(
    private val capacity: Int = DEFAULT_CAPACITY,
) {
    private val lock = Mutex()
    private val byResponse = LinkedHashMap<String, Conversation>()

    suspend fun find(responseId: String): Conversation? =
        lock.withLock {
            // Taken out and put back, which moves it behind every conversation used longer ago.
            byResponse.remove(responseId)?.also { byResponse[responseId] = it }
        }

    suspend fun keep(
        responseId: String,
        conversation: Conversation,
    ) = lock.withLock {
        byResponse[responseId] = conversation
        while (byResponse.size > capacity) byResponse.remove(byResponse.keys.first())
    }

    private companion object {
        const val DEFAULT_CAPACITY = 1024
    }
}
