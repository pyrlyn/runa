package runa.platform

import okio.FileSystem
import okio.IOException
import okio.Path.Companion.toPath
import kotlin.experimental.ExperimentalNativeApi

/**
 * Where the hardware survey gets its facts: the files, calls and programs through which an OS
 * describes the machine. A test hands the survey a machine of its own through this.
 *
 * A source answers null, or with nothing, when this OS does not have it or will not say.
 */
public interface SystemSources {
    public val os: HostOs

    /** `arm64` or `x86_64`. */
    public val architecture: String

    public fun memoryBytes(): Long?

    /** The processors the OS schedules on, hyper-threads included. */
    public fun processorCount(): Int?

    /** The text of a file, for `/proc` and `/sys`. */
    public fun file(path: String): String?

    /** The names in a directory. */
    public fun directory(path: String): List<String>

    /** A text value of the macOS `sysctl`. */
    public fun sysctlText(name: String): String?

    /** A numeric value of the macOS `sysctl`. */
    public fun sysctlNumber(name: String): Long?

    /**
     * What a program prints, or null when it is missing or fails. [command] goes to the shell as
     * it is, so it is never built from anything a user or a file says.
     */
    public fun output(command: String): String?

    /** The GPUs macOS drives through Metal. */
    public fun metalDevices(): List<MetalDevice>
}

/** What Metal says about a GPU, in its own terms. */
public data class MetalDevice(
    val name: String,
    /** `recommendedMaxWorkingSetSize`: what a process may give the GPU before the system slows down. */
    val recommendedMaxWorkingSetSize: Long,
    val hasUnifiedMemory: Boolean,
)

/** This machine. */
public expect fun systemSources(): SystemSources

/** What every OS shares; a source this class does not know answers as if the OS did not have it. */
internal open class HostSources : SystemSources {
    override val os: HostOs get() = hostOs

    @OptIn(ExperimentalNativeApi::class)
    override val architecture: String
        get() =
            when (Platform.cpuArchitecture) {
                CpuArchitecture.ARM64 -> "arm64"
                CpuArchitecture.X64 -> "x86_64"
                else -> Platform.cpuArchitecture.name.lowercase()
            }

    override fun memoryBytes(): Long? = physicalMemoryBytes()

    override fun processorCount(): Int? = null

    override fun file(path: String): String? =
        try {
            FileSystem.SYSTEM.read(path.toPath()) { readUtf8() }
        } catch (e: IOException) {
            null
        }

    override fun directory(path: String): List<String> =
        try {
            FileSystem.SYSTEM.list(path.toPath()).map { it.name }
        } catch (e: IOException) {
            emptyList()
        }

    override fun sysctlText(name: String): String? = null

    override fun sysctlNumber(name: String): Long? = null

    override fun output(command: String): String? = null

    override fun metalDevices(): List<MetalDevice> = emptyList()
}
