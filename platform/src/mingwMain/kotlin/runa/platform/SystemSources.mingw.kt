package runa.platform

import platform.windows.ALL_PROCESSOR_GROUPS
import platform.windows.GetActiveProcessorCount

public actual fun systemSources(): SystemSources = WindowsSources

private object WindowsSources : HostSources() {
    override fun processorCount(): Int? = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS.toUShort()).toInt().takeIf { it > 0 }
}
