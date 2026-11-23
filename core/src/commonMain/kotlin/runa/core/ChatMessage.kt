package runa.core

/** Who wrote a chat message; [id] is the role name of the OpenAI and LM Studio APIs. */
public enum class ChatRole(
    public val id: String,
) {
    SYSTEM("system"),
    USER("user"),
    ASSISTANT("assistant"),
    TOOL("tool"),
}

public data class ChatMessage(
    val role: ChatRole,
    val content: String,
)
