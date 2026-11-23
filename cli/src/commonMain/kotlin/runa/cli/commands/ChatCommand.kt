package runa.cli.commands

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.requireObject
import com.github.ajalt.clikt.core.terminal
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.optional
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.long
import com.github.ajalt.clikt.parameters.types.restrictTo
import com.github.ajalt.mordant.widgets.Text
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
import runa.cli.CommandGroup
import runa.cli.Globals
import runa.cli.daemon.DaemonConnector
import runa.cli.daemon.chooseModel
import runa.cli.daemon.load
import runa.cli.daemon.loadedNote
import runa.cli.output.printError
import runa.client.DaemonClient
import runa.core.BackendError
import runa.core.ModelKey
import runa.core.UserInputError
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.ChatStats
import runa.core.api.LoadRequest

internal class ChatCommand : CliktCommand(name = "chat") {
    private val globals by requireObject<Globals>()
    private val model by argument(
        "model",
        help = "A loaded model from `runa ps` or a model from `runa ls`; without one, the loaded model or a list to choose from",
    ).optional()
    private val prompt by option("-p", "--prompt", metavar = "text", help = "Ask this one question, print the answer and exit")
    private val system by option("-s", "--system", metavar = "text", help = "System prompt")
    private val stats by option("--stats", help = "Print token counts, time to first token and tokens per second after every answer").flag()
    private val ttl by option(
        "--ttl",
        metavar = "seconds",
        help = "When this command loads the model: unload it after this long without a request; 0 for never (default: ttl in config.json)",
    ).long().restrictTo(min = 0)
    private val json by option("--json", help = "With one question: print one JSON object when the answer is complete").flag()

    override val helpTags: Map<String, String> = CommandGroup.LOCAL_MODELS.helpTags

    override fun help(context: Context): String =
        """
        Talk to a model

        The daemon is started when it does not run, and the model is loaded when it is not loaded yet;
        it stays loaded afterwards, see --ttl. The answers are printed as they are generated, and the
        model remembers the conversation until /clear. /help lists what else can be typed. Ctrl-C ends
        the chat, also in the middle of an answer.

        With -p, or with the question on standard input, the one answer goes to standard output and
        nothing else does.
        """.trimIndent()

    override fun run() {
        val question = prompt ?: pipedQuestion()
        if (question == null && globals.wantsJson(json)) throw UserInputError("--json needs one question: give it with -p")
        runBlocking {
            DaemonConnector(globals).require().use { daemon ->
                val instance = instance(daemon.client, model)
                if (question != null) once(daemon.client, instance, question) else converse(daemon.client, instance)
            }
        }
    }

    /** What was piped in, when nobody sits at the keyboard; null in a terminal. */
    private fun pipedQuestion(): String? {
        if (terminal.terminalInfo.inputInteractive) return null
        return generateSequence { terminal.readLineOrNull(false) }.joinToString("\n").trim().ifEmpty {
            throw UserInputError("There is nothing to ask: give the question with -p or on standard input, or run this in a terminal")
        }
    }

    /**
     * The loaded instance that [name] means; the model is loaded first when it is not. Without a
     * name: the one loaded instance, else the model the user picks.
     */
    private suspend fun instance(
        client: DaemonClient,
        name: String?,
    ): String {
        val models = client.models().models
        val loaded = models.flatMap { it.loadedInstances }
        if (name == null) loaded.singleOrNull()?.let { return it.id }
        val forProgram = globals.wantsJson(json)
        val wanted = name ?: terminal.chooseModel(models, mayAsk = !globals.yes && !forProgram)
        loaded.firstOrNull { it.id == wanted }?.let { return it.id }
        val stored = models.singleOrNull { ModelKey.parse(it.key)?.isNamed(wanted) == true }
        stored?.loadedInstances?.firstOrNull()?.let { return it.id }
        // A name that fits no model, or more than one, is the daemon's to explain.
        val response = client.load(terminal, LoadRequest(model = wanted, ttl = ttl), showProgress = !forProgram)
        response.loadedNote()?.let { note -> globals.log.info { note } }
        return checkNotNull(response.instance) { "The daemon loaded a model without saying which" }.id
    }

    private suspend fun once(
        client: DaemonClient,
        instance: String,
        question: String,
    ) {
        val streaming = !globals.wantsJson(json)
        // Nothing follows this answer, so the daemon need not remember it.
        val response = ask(client, ChatState(instance, system), question, streaming, store = false)
        if (!streaming) globals.renderer(json).render(response, ChatResponse.serializer()) { Text(it.text) }
        if (stats) show(response.stats)
    }

    private suspend fun converse(
        client: DaemonClient,
        instance: String,
    ) {
        var state = ChatState(model = instance, systemPrompt = system, showStats = stats)
        globals.log.info { "Chatting with $instance. /help lists the commands, /exit ends the chat" }
        while (true) {
            // On stderr, as a shell's prompt is: stdout holds the answers and nothing else.
            terminal.rawPrint(if (state.pending == null) PROMPT else MORE, stderr = true)
            val line = terminal.readLineOrNull(false)
            if (line == null) {
                // Ctrl-D leaves the cursor behind the prompt.
                terminal.rawPrint("\n", stderr = true)
                return
            }
            val step = state.read(line)
            state = step.state
            when (val action = step.action) {
                ChatAction.Wait -> {}

                ChatAction.Exit -> {
                    return
                }

                is ChatAction.Say -> {
                    terminal.println(action.text, stderr = true)
                }

                is ChatAction.Complain -> {
                    terminal.printError(action.message)
                }

                is ChatAction.Send -> {
                    state = attempt {
                        val response = ask(client, state, action.text, streaming = true, store = true)
                        if (state.showStats) show(response.stats)
                        state.answered(response)
                    } ?: state
                }

                is ChatAction.UseModel -> {
                    state = attempt {
                        state
                            .copy(
                                model = instance(client, action.name),
                            ).also { terminal.println("The model is ${it.model}", stderr = true) }
                    } ?: state
                }
            }
        }
    }

    /** A mistake or a failure ends one answer, not the chat: null when [block] failed. */
    private suspend fun <T> attempt(block: suspend () -> T): T? =
        try {
            block()
        } catch (e: UserInputError) {
            terminal.printError(e.message)
            null
        } catch (e: BackendError) {
            terminal.printError(e.message)
            null
        }

    private suspend fun ask(
        client: DaemonClient,
        state: ChatState,
        text: String,
        streaming: Boolean,
        store: Boolean,
    ): ChatResponse {
        val request =
            ChatRequest(
                model = state.model,
                input = JsonPrimitive(text),
                // Always said, so the daemon never falls back on what an earlier turn had.
                systemPrompt = state.systemPrompt.orEmpty(),
                store = store,
                previousResponseId = state.previousResponseId,
            )
        var endsLine = true
        try {
            return client.chat(request) { piece ->
                if (streaming && piece.isNotEmpty()) {
                    terminal.rawPrint(piece)
                    endsLine = piece.endsWith('\n')
                }
            }
        } finally {
            // Whatever comes next, the prompt or an error, must start on a line of its own.
            if (!endsLine) terminal.rawPrint("\n")
        }
    }

    private fun show(stats: ChatStats) = terminal.println(terminal.theme.muted(stats.summary()), stderr = true)

    private companion object {
        const val PROMPT = "> "
        const val MORE = ". "
    }
}
