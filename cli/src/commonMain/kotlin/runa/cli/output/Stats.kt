package runa.cli.output

/** The one line of `--stats`, the same for every command that generates text. */
internal fun statsSummary(
    promptTokens: Int,
    answerTokens: Int,
    /** `stop` or `length`. */
    stopReason: String,
    secondsToFirstToken: Double,
    tokensPerSecond: Double,
): String =
    "prompt: $promptTokens tokens, answer: $answerTokens tokens ($stopReason), " +
        "time to first token: $secondsToFirstToken s, $tokensPerSecond tokens/s"
