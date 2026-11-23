package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.getenv
import kotlin.experimental.ExperimentalNativeApi

/** The operating systems runa runs on; decides where its files live. */
public enum class HostOs { MACOS, LINUX, WINDOWS }

/** The OS this binary was built for. */
@OptIn(ExperimentalNativeApi::class)
public val hostOs: HostOs =
    when (Platform.osFamily) {
        OsFamily.MACOSX -> HostOs.MACOS
        OsFamily.WINDOWS -> HostOs.WINDOWS
        else -> HostOs.LINUX
    }

/** Reads a process environment variable; null when unset. */
@OptIn(ExperimentalForeignApi::class)
public fun systemEnv(name: String): String? = getenv(name)?.toKString()
