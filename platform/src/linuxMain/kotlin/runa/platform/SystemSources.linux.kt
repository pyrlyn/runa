package runa.platform

public actual fun systemSources(): SystemSources = LinuxSources

private object LinuxSources : UnixSources()
