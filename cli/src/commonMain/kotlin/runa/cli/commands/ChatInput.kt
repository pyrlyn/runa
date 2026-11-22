package runa.cli.commands

import runa.cli.output.statsSummary
import runa.core.api.ChatResponse
import runa.core.api.ChatStats

// What `runa chat` makes of the lines typed at its prompt. Nothing here reads, prints or talks to
// the daemon: a line and the state before it give the state after it and what to do, so every rule
// of the input can be tested without a terminal.

/** What `runa chat` knows between two lines of input. */
internal data class ChatState(
    /** The loaded instance that answers. */
    val model: String,
    val systemPrompt: String? = null,
    /** True to print the statistics of every answer. */
    val showStats: Boolean = false,
    /** The answer that the next message continues; null at the start of a conversation. */
    val previousResponseId: String? = null,
    val lastStats: ChatStats? = null,
    /** The message that is being typed over several lines; null between messages. */
    val pending: PendingMessage? = null,
) {
    /** The line was read: this is the state after it and what there is to do. */
    fun read(line: String): ChatStep {
        pending?.let { return it.continued(line) }
        val text = line.trim()
        return when {
            text.isEmpty() -> ChatStep(this, ChatAction.Wait)
            text.startsWith(BLOCK) -> PendingMessage(emptyList(), block = true).continued(text.removePrefix(BLOCK))
            line.endsWith(CONTINUED) -> PendingMessage(emptyList(), block = false).continued(line)
            text.startsWith(COMMAND) -> command(text.substringBefore(' '), text.substringAfter(' ', "").trim())
            else -> ChatStep(this, ChatAction.Send(text))
        }
    }

    /** The model answered the message that [read] said to send. */
    fun answered(response: ChatResponse): ChatState = copy(previousResponseId = response.responseId, lastStats = response.stats)

    private fun PendingMessage.continued(line: String): ChatStep {
        val closed = if (block) line.trimEnd().endsWith(BLOCK) else !line.endsWith(CONTINUED)
        val content =
            when {
                block && closed -> line.trimEnd().removeSuffix(BLOCK)
                block || closed -> line
                else -> line.removeSuffix(CONTINUED)
            }
        val lines = lines + content
        if (!closed) return ChatStep(copy(pending = copy(lines = lines)), ChatAction.Wait)
        val text = lines.joinToString("\n").trim()
        return ChatStep(copy(pending = null), if (text.isEmpty()) ChatAction.Wait else ChatAction.Send(text))
    }

    private fun command(
        name: String,
        argument: String,
    ): ChatStep {
        fun say(
            text: String,
            state: ChatState = this,
        ) = ChatStep(state, ChatAction.Say(text))
        return when (name) {
            "/help" -> {
                say(HELP)
            }

            "/exit" -> {
                ChatStep(this, ChatAction.Exit)
            }

            "/clear" -> {
                say("Started a new conversation", copy(previousResponseId = null))
            }

            "/model" -> {
                if (argument.isEmpty()) say("The model is $model") else ChatStep(this, ChatAction.UseModel(argument))
            }

            "/system-prompt" -> {
                when {
                    argument.isNotEmpty() -> say("Set the system prompt", copy(systemPrompt = argument))
                    systemPrompt == null -> say("There is no system prompt")
                    else -> say("Removed the system prompt", copy(systemPrompt = null))
                }
            }

            "/stats" -> {
                when (argument) {
                    "" -> say(lastStats?.summary() ?: "There is no answer yet")
                    "on" -> say("Statistics follow every answer", copy(showStats = true))
                    "off" -> say("Statistics are off", copy(showStats = false))
                    else -> ChatStep(this, ChatAction.Complain("/stats takes on, off or nothing"))
                }
            }

            else -> {
                ChatStep(this, ChatAction.Complain("$name is not a command; /help lists them"))
            }
        }
    }

    companion object {
        /** Opens and closes a message of several lines. */
        const val BLOCK = "\"\"\""

        /** At the end of a line: the message goes on in the next one. */
        const val CONTINUED = "\\"
        const val COMMAND = "/"

        val HELP =
            """
            /help                  Show this list
            /exit                  End the chat; so does Ctrl-D
            /clear                 Start a new conversation
            /model [name]          Show the model, or switch to another one
            /system-prompt [text]  Set the system prompt; without text, remove it
            /stats [on|off]        Show the statistics of the last answer, or of every answer
            $BLOCK                    Start a message of several lines; $BLOCK ends it
            $CONTINUED                      At the end of a line: the message continues on the next one
            """.trimIndent()
    }
}

/** The lines so far of a message that is not complete. */
internal data class PendingMessage(
    val lines: List<String>,
    /** True between two [ChatState.BLOCK] marks; false for a line that ended in [ChatState.CONTINUED]. */
    val block: Boolean,
)

internal data class ChatStep(
    val state: ChatState,
    val action: ChatAction,
)

/** What `runa chat` does after a line of input. */
internal sealed interface ChatAction {
    /** Nothing but reading the next line. */
    data object Wait : ChatAction

    /** Ask the model. */
    data class Send(
        val text: String,
    ) : ChatAction

    /** Tell the user. */
    data class Say(
        val text: String,
    ) : ChatAction

    /** Tell the user what was wrong with the line. */
    data class Complain(
        val message: String,
    ) : ChatAction

    /** Switch to the model the user named, loading it when needed. */
    data class UseModel(
        val name: String,
    ) : ChatAction

    data object Exit : ChatAction
}

/** The line of `--stats` and `/stats`. */
internal fun ChatStats.summary(): String =
    statsSummary(inputTokens, totalOutputTokens, stopReason, timeToFirstTokenSeconds, tokensPerSecond)
