package runa.engine.llama

import okio.FileSystem
import okio.Path.Companion.toPath

// Kotlin/Native's Linux bindings have no getrusage; the kernel reports the same number here.
internal actual fun peakResidentSize(): Long =
    FileSystem.SYSTEM
        .read("/proc/self/status".toPath()) { readUtf8() }
        .lineSequence()
        .first { it.startsWith(PEAK) }
        .removePrefix(PEAK)
        .trim()
        .substringBefore(' ')
        .toLong()

private const val PEAK = "VmHWM:"
