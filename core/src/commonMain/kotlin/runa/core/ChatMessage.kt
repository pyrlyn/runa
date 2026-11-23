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
    /** The tools an assistant's message asked for. */
    val toolCalls: List<ToolCall> = emptyList(),
    /** In a tool's message: the [ToolCall.id] it answers. */
    val toolCallId: String? = null,
)

/**
 * A model's request to run a tool.
 *
 * @property arguments JSON as the model wrote it: it is meant to be an object, but nothing makes a
 * model write valid JSON unless the tool's parameters constrained it.
 */
public data class ToolCall(
    val id: String,
    val name: String,
    val arguments: String,
)
