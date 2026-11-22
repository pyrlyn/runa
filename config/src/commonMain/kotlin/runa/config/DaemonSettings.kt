package runa.config

import okio.Path
import runa.core.GpuOffload
import runa.core.LogLevel
import runa.platform.RunaPaths
import kotlin.time.Duration

/** What the daemon process is started with: the settings of the command line that started it. */
public data class DaemonSettings(
    val version: String,
    val paths: RunaPaths,
    val modelsDir: Path,
    /** The loopback port to listen on; null for any free one. */
    val port: Int?,
    val logLevel: LogLevel,
    /** `context_length` of `config.json`; null for the built-in default. */
    val contextLength: Int?,
    val gpu: GpuOffload,
    /** `ttl` of `config.json`; null for never. */
    val ttl: Duration?,
)
