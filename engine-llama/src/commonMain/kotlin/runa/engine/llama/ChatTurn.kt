package runa.engine.llama

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import runa.core.ChatMessage
import runa.core.UserInputError
import runa.engine.GenerationEvent
import runa.engine.Prompt
import runa.engine.ResponseFormat
import runa.engine.Tool
import runa.engine.ToolChoice

// The JSON that crosses the C API of the shim (nativeInterop/shim/runa_llama.h). The shim speaks
// JSON because llama.cpp's own chat code does: messages and tools go in as OpenAI words them.

/** The `request` of `runa_turn_init`. */
internal fun chatRequest(chat: Prompt.Chat): JsonObject {
    if (chat.messages.isEmpty()) throw UserInputError("A chat needs at least one message")
    // llama.cpp can require a call but not name it. With only the named tool offered, a required
    // call is that tool's.
    val tools = chat.offeredTools()
    return buildJsonObject {
        putJsonArray("messages") { chat.messages.forEach { add(messageJson(it)) } }
        if (tools.isNotEmpty()) {
            putJsonArray("tools") { tools.forEach { add(toolJson(it)) } }
            put("tool_choice", if (chat.toolChoice == ToolChoice.Auto) "auto" else "required")
            chat.parallelToolCalls?.let { put("parallel_tool_calls", it) }
        }
        when (val format = chat.format) {
            ResponseFormat.Text -> Unit
            is ResponseFormat.Json -> put("json_schema", format.schema ?: ANY_OBJECT)
        }
    }
}

private fun messageJson(message: ChatMessage): JsonObject =
    buildJsonObject {
        put("role", message.role.id)
        put("content", message.content)
        if (message.toolCalls.isNotEmpty()) {
            putJsonArray("tool_calls") {
                message.toolCalls.forEach { call ->
                    add(
                        buildJsonObject {
                            put("id", call.id)
                            put("type", "function")
                            putJsonObject("function") {
                                put("name", call.name)
                                put("arguments", call.arguments)
                            }
                        },
                    )
                }
            }
        }
        message.toolCallId?.let { put("tool_call_id", it) }
    }

private fun toolJson(tool: Tool): JsonObject =
    buildJsonObject {
        put("type", "function")
        putJsonObject("function") {
            put("name", tool.name)
            tool.description?.let { put("description", it) }
            tool.parameters?.let { put("parameters", it) }
        }
    }

private val ANY_OBJECT = buildJsonObject { put("type", "object") }

/** What `runa_turn_start` returns: the rendered chat and the strings that end the answer. */
internal class TurnStart(
    val prompt: String,
    val stops: List<String>,
)

internal fun turnStart(json: String): TurnStart {
    val start = Json.parseToJsonElement(json).jsonObject
    return TurnStart(
        prompt = start.getValue("prompt").jsonPrimitive.content,
        stops = start.getValue("stops").jsonArray.map { it.jsonPrimitive.content },
    )
}

/** What `runa_turn_accept` and `runa_turn_finish` return. */
internal fun turnEvents(json: String): List<GenerationEvent> =
    Json.parseToJsonElement(json).jsonArray.map { element ->
        val (kind, value) = element.jsonObject.entries.single()
        when (kind) {
            "content" -> {
                GenerationEvent.Text(value.jsonPrimitive.content)
            }

            "call" -> {
                val call = value.jsonObject
                GenerationEvent.ToolCallStarted(
                    index = call.getValue("index").jsonPrimitive.int,
                    id = call.getValue("id").jsonPrimitive.content,
                    name = call.getValue("name").jsonPrimitive.content,
                )
            }

            "arguments" -> {
                val arguments = value.jsonObject
                GenerationEvent.ToolCallArguments(
                    index = arguments.getValue("index").jsonPrimitive.int,
                    text = arguments.getValue("text").jsonPrimitive.content,
                )
            }

            else -> {
                error("The shim reported an unknown event: $kind")
            }
        }
    }
