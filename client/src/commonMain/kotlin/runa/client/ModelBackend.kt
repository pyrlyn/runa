package runa.client

import runa.core.BackendError
import runa.core.UserInputError
import runa.core.api.ChatRequest
import runa.core.api.ChatResponse
import runa.core.api.LoadRequest
import runa.core.api.LoadResponse
import runa.core.api.ModelList
import runa.core.api.UnloadRequest
import runa.core.api.UnloadResponse

/**
 * What the model commands need from the service that holds the models: runa's own daemon
 * ([DaemonClient]) or a running LM Studio ([LmStudioBackend]). Both answer in the types of runa's
 * API; what a service does not tell is null there.
 *
 * Every call throws [UserInputError] when the service says the request is wrong, and [BackendError]
 * when it cannot be reached, rejects the token or fails.
 */
public interface ModelBackend : AutoCloseable {
    /** Every model the service has, each with its loaded instances. */
    public suspend fun models(): ModelList

    /** Loads a model and returns when it is ready; [onProgress] gets values from 0 to 1 when the service reports any. */
    public suspend fun load(
        request: LoadRequest,
        onProgress: (Float) -> Unit = {},
    ): LoadResponse

    /**
     * Sends one turn of a conversation and returns the whole answer; [onText] gets its pieces as
     * the model writes them. Cancelling the call ends the generation.
     */
    public suspend fun chat(
        request: ChatRequest,
        onText: (String) -> Unit = {},
    ): ChatResponse

    public suspend fun unload(request: UnloadRequest): UnloadResponse
}
