package runa.cli.output

import kotlin.math.roundToLong

/** A file size in decimal units, as download pages and `lms` show it: `4.68 GB`. */
internal fun formatBytes(bytes: Long): String = scaled(bytes, listOf(" B", " KB", " MB", " GB", " TB"))

/** A count such as a model's parameters: `7.62B`, `494M`. */
internal fun formatCount(count: Long): String = scaled(count, listOf("", "K", "M", "B", "T"))

/** A time span in its largest whole unit, as `ps` shows idle times: `45s`, `12m`, `3h`, `2d`. */
internal fun formatSeconds(seconds: Long): String =
    when {
        seconds < SECONDS_PER_MINUTE -> "${seconds}s"
        seconds < SECONDS_PER_HOUR -> "${seconds / SECONDS_PER_MINUTE}m"
        seconds < SECONDS_PER_DAY -> "${seconds / SECONDS_PER_HOUR}h"
        else -> "${seconds / SECONDS_PER_DAY}d"
    }

private const val SECONDS_PER_MINUTE = 60L
private const val SECONDS_PER_HOUR = 3_600L
private const val SECONDS_PER_DAY = 86_400L

// Three significant digits: enough to tell two quantizations apart, short enough for a column.
private fun scaled(
    value: Long,
    units: List<String>,
): String {
    var scaled = value.toDouble()
    var unit = 0
    // 999.5 and up would round to four digits, so it moves to the next unit instead.
    while (scaled >= 999.5 && unit < units.lastIndex) {
        scaled /= 1000
        unit++
    }
    val decimals =
        when {
            unit == 0 -> 0
            scaled < 9.995 -> 2
            scaled < 99.95 -> 1
            else -> 0
        }
    return fixed(scaled, decimals) + units[unit]
}

// Kotlin's common library has no number formatting, and this is all the CLI needs of it.
private fun fixed(
    value: Double,
    decimals: Int,
): String {
    if (decimals == 0) return value.roundToLong().toString()
    val factor = if (decimals == 1) 10 else 100
    val rounded = (value * factor).roundToLong()
    return "${rounded / factor}.${(rounded % factor).toString().padStart(decimals, '0')}"
}
