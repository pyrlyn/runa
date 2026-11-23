package runa.platform

import okio.Path
import okio.Path.Companion.toPath

/**
 * Where runa keeps its files (PLAN.md, section 7). The three directories are the same one
 * (`~/.runa`) unless XDG variables split them on Linux.
 */
public data class RunaPaths(
    /** `config.json`. */
    val configDir: Path,
    /** The model store. */
    val dataDir: Path,
    /** Discovery file, logs and CLI preferences. */
    val stateDir: Path,
) {
    val configFile: Path get() = configDir / "config.json"
    val modelsDir: Path get() = dataDir / "models"

    /** `{pid, port, token, version}` of the running daemon; how the CLI finds it. */
    val discoveryFile: Path get() = stateDir / "run" / "daemon.json"

    /** Held by the daemon for as long as it runs, so there is never a second one. */
    val daemonLockFile: Path get() = stateDir / "run" / "daemon.lock"
    val daemonLogFile: Path get() = stateDir / "logs" / "daemon.log"
}

/**
 * `RUNA_HOME` overrides everything. Otherwise Linux honours each set `XDG_*_HOME`, Windows uses
 * `%LOCALAPPDATA%\runa`, and the fallback everywhere is `~/.runa`.
 */
public fun resolveRunaPaths(
    os: HostOs,
    env: (String) -> String?,
): RunaPaths {
    fun dir(name: String): Path? = env(name)?.takeIf { it.isNotBlank() }?.toPath()

    dir("RUNA_HOME")?.let { return RunaPaths(it, it, it) }

    val home by lazy {
        val base =
            dir(if (os == HostOs.WINDOWS) "USERPROFILE" else "HOME")
                ?: throw IllegalStateException("Cannot find the home directory: set HOME or RUNA_HOME")
        base / ".runa"
    }
    return when (os) {
        HostOs.MACOS -> {
            RunaPaths(home, home, home)
        }

        HostOs.WINDOWS -> {
            (dir("LOCALAPPDATA")?.div("runa") ?: home).let { RunaPaths(it, it, it) }
        }

        HostOs.LINUX -> {
            // The XDG spec says a relative path in these variables is invalid and must be ignored.
            fun xdg(name: String): Path = dir(name)?.takeIf { it.isAbsolute }?.div("runa") ?: home
            RunaPaths(xdg("XDG_CONFIG_HOME"), xdg("XDG_DATA_HOME"), xdg("XDG_STATE_HOME"))
        }
    }
}
