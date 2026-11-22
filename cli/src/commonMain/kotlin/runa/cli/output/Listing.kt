package runa.cli.output

import com.github.ajalt.mordant.rendering.TextStyles
import com.github.ajalt.mordant.rendering.Widget
import com.github.ajalt.mordant.table.Borders
import com.github.ajalt.mordant.table.table

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
