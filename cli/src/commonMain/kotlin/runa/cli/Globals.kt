package runa.cli

import com.github.ajalt.mordant.terminal.Terminal
import okio.Path.Companion.toPath
import runa.cli.output.Diagnostics
import runa.cli.output.HumanRenderer
import runa.cli.output.JsonRenderer
import runa.cli.output.Renderer
import runa.config.ConfigFile
import runa.config.ConfigKey
import runa.config.ConfigResolver
import runa.core.LogLevel
import runa.hub.HubClient
import runa.platform.RunaPaths
import runa.platform.resolveRunaPaths
import runa.store.ModelStore

/** The root command's parsed global options, shared with every subcommand through the Clikt context. */
internal class Globals(
    val environment: CliEnvironment,
    val terminal: Terminal,
    private val json: Boolean,
    private val quiet: Boolean,
    private val verbose: Boolean,
    /** `-y`: answer every prompt with its default. Read by the interactive commands of later phases. */
    val yes: Boolean,
    private val flags: Map<ConfigKey, String>,
) {
    val paths: RunaPaths = resolveRunaPaths(environment.os, environment.env)

    /** Reads `config.json` from disk on every call, so a command sees its own `config set`. */
    fun loadConfigFile(): ConfigFile = ConfigFile.load(environment.fileSystem, paths.configFile)

    // The file is read on first use: `version` and `config path` keep working when it is broken.
    val config: ConfigResolver = ConfigResolver(flags, environment.env, ::loadConfigFile, paths)

    val log: Diagnostics =
        Diagnostics(terminal) {
            when {
                quiet -> LogLevel.ERROR
                verbose -> LogLevel.DEBUG
                else -> LogLevel.parse(config.resolve(ConfigKey.LOG_LEVEL).value.orEmpty()) ?: LogLevel.INFO
            }
        }

    /** The model store at the configured `models_dir`. */
    val store: ModelStore by lazy {
        ModelStore(environment.fileSystem, checkNotNull(config.resolve(ConfigKey.MODELS_DIR).value).toPath())
    }

    /** A client for Hugging Face, or for the mirror named by `HF_ENDPOINT`; the caller closes it. */
    fun hub(): HubClient {
        fun env(name: String) = environment.env(name)?.takeIf { it.isNotBlank() }
        return HubClient(environment.httpEngine, env("HF_ENDPOINT") ?: HubClient.DEFAULT_ENDPOINT, env("HF_TOKEN"))
    }

    /** [jsonHere] is a command's own `--json`, so both `runa --json ls` and `runa ls --json` work. */
    fun wantsJson(jsonHere: Boolean = false): Boolean = json || jsonHere

    fun renderer(jsonHere: Boolean = false): Renderer = if (wantsJson(jsonHere)) JsonRenderer(terminal) else HumanRenderer(terminal)
}
