package runa.cli

import com.github.ajalt.mordant.terminal.Terminal
import okio.Path.Companion.toPath
import runa.cli.output.Diagnostics
import runa.cli.output.HumanRenderer
import runa.cli.output.JsonRenderer
import runa.cli.output.Renderer
import runa.config.ConfigFile
import runa.config.ConfigKey
import runa.config.ConfigKind
import runa.config.ConfigResolver
import runa.config.DaemonSettings
import runa.core.GpuOffload
import runa.core.LogLevel
import runa.engine.Engine
import runa.hub.HubClient
import runa.platform.RunaPaths
import runa.platform.resolveRunaPaths
import runa.store.ModelStore
import kotlin.time.Duration.Companion.seconds

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
    tokenFlag: String? = null,
) {
    val paths: RunaPaths = resolveRunaPaths(environment.os, environment.env)

    /** Reads `config.json` from disk on every call, so a command sees its own `config set`. */
    fun loadConfigFile(): ConfigFile = ConfigFile.load(environment.fileSystem, paths.configFile)

    // The file is read on first use: `version` and `config path` keep working when it is broken.
    val config: ConfigResolver = ConfigResolver(flags, environment.env, ::loadConfigFile, paths)

    private val logLevel: LogLevel by lazy {
        when {
            quiet -> LogLevel.ERROR
            verbose -> LogLevel.DEBUG
            else -> LogLevel.parse(config.resolve(ConfigKey.LOG_LEVEL).value.orEmpty()) ?: LogLevel.INFO
        }
    }

    val log: Diagnostics = Diagnostics(terminal) { logLevel }

    /** The bearer token for a daemon whose discovery file this machine does not have. */
    val token: String? = (tokenFlag ?: environment.env(TOKEN_ENV))?.takeIf { it.isNotBlank() }

    /** The model store at the configured `models_dir`. */
    val store: ModelStore by lazy {
        ModelStore(environment.fileSystem, checkNotNull(config.resolve(ConfigKey.MODELS_DIR).value).toPath())
    }

    /** The in-process engine of `runa run`, created on first use: no other command pays for it. */
    val engine: Engine by lazy { environment.engine { level, line -> log.at(level) { line } } }

    /** A client for Hugging Face, or for the mirror named by `HF_ENDPOINT`; the caller closes it. */
    fun hub(): HubClient {
        fun env(name: String) = environment.env(name)?.takeIf { it.isNotBlank() }
        return HubClient(environment.httpEngine, env("HF_ENDPOINT") ?: HubClient.DEFAULT_ENDPOINT, env("HF_TOKEN"))
    }

    /** What a daemon started by this command line runs with. */
    fun daemonSettings(): DaemonSettings =
        DaemonSettings(
            version = environment.version,
            paths = paths,
            modelsDir = store.modelsDir,
            port = config.resolve(ConfigKey.PORT).value?.toInt(),
            logLevel = logLevel,
            logPrompts = config.resolve(ConfigKey.LOG_PROMPTS).value == "true",
            contextLength = config.resolve(ConfigKey.CONTEXT_LENGTH).value?.toInt(),
            gpu = GpuOffload.parse(config.resolve(ConfigKey.GPU).value.orEmpty()) ?: GpuOffload.Max,
            ttl =
                config
                    .resolve(ConfigKey.TTL)
                    .value
                    ?.toInt()
                    ?.seconds,
        )

    /** The global flags among [keys] that were given, as arguments for another `runa` process. */
    fun flagArguments(vararg keys: ConfigKey): List<String> =
        keys.flatMap { key ->
            val value = flags[key]
            when {
                value == null -> emptyList()

                // A switch has no value: it is there or it is not.
                key.kind == ConfigKind.BOOL -> listOfNotNull(key.flag.takeIf { value == "true" })

                else -> listOf(key.flag, value)
            }
        }

    /** [jsonHere] is a command's own `--json`, so both `runa --json ls` and `runa ls --json` work. */
    fun wantsJson(jsonHere: Boolean = false): Boolean = json || jsonHere

    fun renderer(jsonHere: Boolean = false): Renderer = if (wantsJson(jsonHere)) JsonRenderer(terminal) else HumanRenderer(terminal)

    companion object {
        const val TOKEN_ENV: String = "RUNA_TOKEN"
    }
}
