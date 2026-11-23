package runa.platform

/**
 * How many threads compute-bound work should use: the physical cores, and on Apple silicon only
 * the performance ones. Hyper-threads and efficiency cores slow matrix code down. At least 1.
 */
public fun computeCoreCount(): Int = surveyCpu(systemSources()).let { it.performanceCores ?: it.cores ?: it.threads } ?: 1
