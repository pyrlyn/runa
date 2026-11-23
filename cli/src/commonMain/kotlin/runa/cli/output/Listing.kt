package runa.cli.output

import com.github.ajalt.mordant.rendering.TextStyles
import com.github.ajalt.mordant.rendering.Widget
import com.github.ajalt.mordant.table.Borders
import com.github.ajalt.mordant.table.table
import com.github.ajalt.mordant.terminal.Terminal
import com.github.ajalt.mordant.terminal.prompt

/**
 * The one table style of runa's listings: no borders, a bold header, two spaces between columns,
 * so rows stay easy to read and to cut with `awk`.
 */
internal fun listing(
    headers: List<String>,
    rows: List<List<String>>,
): Widget =
    table {
        tableBorders = Borders.NONE
        cellBorders = Borders.NONE
        padding { right = 2 }
        header {
            style = TextStyles.bold.style
            row { cellsFrom(headers) }
        }
        body { rows.forEach { row { cellsFrom(it) } } }
    }

/**
 * Shows [rows] as a numbered list and asks which one; Enter takes [default]. Returns the row's
 * index, or null when the input ends without an answer.
 */
internal fun Terminal.pick(
    label: String,
    headers: List<String>,
    rows: List<List<String>>,
    default: Int = 0,
): Int? {
    println(listing(listOf("#") + headers, rows.mapIndexed { index, row -> listOf("${index + 1}") + row }))
    return prompt(label, default = "${default + 1}", showChoices = false, choices = rows.indices.map { "${it + 1}" })?.toInt()?.minus(1)
}
