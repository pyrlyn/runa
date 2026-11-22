package runa.core

/** The user asked for something invalid. Printed as one line, no stack trace; exit [ExitCode.USER_INPUT]. */
public class UserInputError(
    message: String,
) : RuntimeException(message)

/** The daemon or a remote service could not be reached or failed. Exit [ExitCode.BACKEND]. */
public class BackendError(
    message: String,
    cause: Throwable? = null,
) : RuntimeException(message, cause)

/** Process exit codes of the `runa` executable. */
public object ExitCode {
    public const val OK: Int = 0
    public const val UNEXPECTED: Int = 1
    public const val USER_INPUT: Int = 2
    public const val BACKEND: Int = 3

    /** 128 + SIGINT, the shell convention for a process stopped by Ctrl-C. */
    public const val INTERRUPTED: Int = 130
}
