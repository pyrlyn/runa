package runa.platform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.staticCFunction
import platform.posix.SIGINT
import platform.posix._exit
import platform.posix.signal

/**
 * Makes Ctrl-C end the process with [exitCode] instead of the signal's default action, so shells
 * and scripts see a normal exit status. `_exit` because only async-signal-safe calls are allowed
 * inside a signal handler.
 */
@OptIn(ExperimentalForeignApi::class)
public fun exitOnInterrupt(exitCode: Int) {
    interruptExitCode = exitCode
    signal(SIGINT, staticCFunction<Int, Unit> { _exit(interruptExitCode) })
}

// A static C function cannot capture, so the handler reads the code from here.
private var interruptExitCode: Int = 130
