package runa.core

/** How to load a model. A null field leaves the choice to the model or the engine. */
public data class LoadConfig(
    /** Tokens of context; null means the model's own context length. */
    val contextLength: Int? = null,
    val gpu: GpuOffload = GpuOffload.Max,
    /** CPU threads for inference; null means the engine picks. */
    val threads: Int? = null,
)
