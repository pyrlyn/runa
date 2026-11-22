package runa.cli.output

import com.github.ajalt.mordant.rendering.Widget
import com.github.ajalt.mordant.terminal.Terminal
import kotlinx.serialization.SerializationStrategy
import kotlinx.serialization.json.Json

/**
 * Prints a command's result. Commands describe the value once, as data plus a human view, and
 * never check for `--json` themselves.
 */
internal sealed interface Renderer {
    fun <T> render(
        value: T,
        serializer: SerializationStrategy<T>,
        human: (T) -> Widget,
    )
}

/**
 * Tables and colours. Mordant drops the colours when stdout is not a TTY or `NO_COLOR` is set.
 * Table cells are padded to the column width, so the padding after the last column is cut.
 */
internal class HumanRenderer(
    private val terminal: Terminal,
) : Renderer {
    override fun <T> render(
        value: T,
        serializer: SerializationStrategy<T>,
        human: (T) -> Widget,
    ) = terminal.rawPrint(terminal.render(human(value)).lineSequence().joinToString("\n") { it.trimEnd() } + "\n")
}

/** One line of JSON with the stable field names of the `@SerialName`s; never styled or wrapped. */
internal class JsonRenderer(
    private val terminal: Terminal,
) : Renderer {
    override fun <T> render(
        value: T,
        serializer: SerializationStrategy<T>,
        human: (T) -> Widget,
    ) = terminal.rawPrint(outputJson.encodeToString(serializer, value) + "\n")
}

// Defaults and nulls are written out: scripts should see every field on every run.
internal val outputJson: Json = Json { encodeDefaults = true }
