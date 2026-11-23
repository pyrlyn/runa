package runa.platform

import platform.windows.ALL_PROCESSOR_GROUPS
import platform.windows.GetActiveProcessorCount

// Logical processors: telling cores from hyper-threads needs GetLogicalProcessorInformationEx,
// which is not worth its code until Windows is a supported target (PLAN.md, section 13).
public actual fun computeCoreCount(): Int = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS.toUShort()).toInt().coerceAtLeast(1)
