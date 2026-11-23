package runa.platform

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.LongVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.pointed
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.sizeOf
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value
import platform.Metal.MTLCopyAllDevices
import platform.Metal.MTLDeviceProtocol
import platform.darwin.sysctlbyname
import platform.posix.size_tVar

public actual fun systemSources(): SystemSources = AppleSources

@OptIn(ExperimentalForeignApi::class)
private object AppleSources : UnixSources() {
    override fun sysctlText(name: String): String? =
        memScoped {
            val size = alloc<size_tVar>()
            if (sysctlbyname(name, null, size.ptr, null, 0u) != 0 || size.value.toLong() !in 1..MAX_TEXT_BYTES) return null
            // One more than asked for, so the text ends even if the kernel did not end it.
            val buffer = allocArray<ByteVar>(size.value.toInt() + 1)
            if (sysctlbyname(name, buffer, size.ptr, null, 0u) != 0) return null
            buffer.toKString().trim().takeIf { it.isNotEmpty() }
        }

    /** The kernel has both 32-bit and 64-bit numbers and says which by the size it writes. */
    override fun sysctlNumber(name: String): Long? =
        memScoped {
            val value = alloc<LongVar>()
            value.value = 0
            val size = alloc<size_tVar>()
            size.value = sizeOf<LongVar>().convert()
            if (sysctlbyname(name, value.ptr, size.ptr, null, 0u) != 0) return null
            when (size.value.toLong()) {
                sizeOf<IntVar>() -> {
                    value.ptr
                        .reinterpret<IntVar>()
                        .pointed.value
                        .toLong()
                }

                sizeOf<LongVar>() -> {
                    value.value
                }

                else -> {
                    null
                }
            }
        }

    // All of them, and not the default one: that one needs a window server on some machines.
    override fun metalDevices(): List<MetalDevice> =
        MTLCopyAllDevices().filterIsInstance<MTLDeviceProtocol>().map {
            MetalDevice(it.name, it.recommendedMaxWorkingSetSize.toLong(), it.hasUnifiedMemory)
        }

    private const val MAX_TEXT_BYTES = 4096L
}
