package runa.cli.output

/** The one line of `--stats`, the same for every command that generates text. */
internal fun statsSummary(
    promptTokens: Int,
    answerTokens: Int,
    /** `stop` or `length`; null when the service does not tell. */
    stopReason: String?,
    secondsToFirstToken: Double,
    tokensPerSecond: Double,
): String =
    // runa's own numbers arrive rounded; LM Studio's come with every digit of a double.
    "prompt: $promptTokens tokens, answer: $answerTokens tokens${stopReason?.let { " ($it)" }.orEmpty()}, " +
        "time to first token: ${hundredths(secondsToFirstToken)} s, ${hundredths(tokensPerSecond)} tokens/s"
