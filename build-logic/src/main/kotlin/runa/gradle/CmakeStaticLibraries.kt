package runa.gradle

import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.Internal
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations
import org.gradle.work.DisableCachingByDefault
import org.jetbrains.kotlin.konan.target.AppleConfigurables
import org.jetbrains.kotlin.konan.target.Distribution
import org.jetbrains.kotlin.konan.target.Family
import org.jetbrains.kotlin.konan.target.GccConfigurables
import org.jetbrains.kotlin.konan.target.KonanTarget
import org.jetbrains.kotlin.konan.target.Platform
import org.jetbrains.kotlin.konan.target.PlatformManager
import java.io.File
import java.nio.file.Files
import javax.inject.Inject

/**
 * Builds the static libraries of a vendored CMake project for one Kotlin/Native target, so that
 * cinterop can embed them into the klib.
 *
 * Linux targets are compiled with the clang and the sysroot Kotlin/Native itself links against
 * (glibc 2.19 and the libstdc++ of GCC 8): objects from the host compiler expect a newer
 * libstdc++ and do not link there. That toolchain cross-compiles, so any host builds both Linux
 * architectures. macOS uses Xcode's toolchain, as Kotlin/Native does.
 */
@DisableCachingByDefault(because = "CMake keeps its own incremental state in the build tree")
abstract class CmakeStaticLibraries
    @Inject
    constructor(
        private val exec: ExecOperations,
    ) : DefaultTask() {
        /** The directory with the top-level `CMakeLists.txt`. */
        @get:Internal
        abstract val sourceDir: DirectoryProperty

        /** What stands for the content of [sourceDir] in up-to-date checks: the pinned commit. */
        @get:Input
        abstract val sourceRevision: Property<String>

        /** A [KonanTarget] name, for example `linux_x64`. */
        @get:Input
        abstract val konanTarget: Property<String>

        /** `-D` options of the project, without the toolchain ones this task adds. */
        @get:Input
        abstract val options: ListProperty<String>

        /** The CMake targets to build. */
        @get:Input
        abstract val buildTargets: ListProperty<String>

        /** The Kotlin/Native distribution; its `konan.properties` name the compiler and sysroot. */
        @get:Input
        abstract val konanHome: Property<String>

        @get:Input
        @get:Optional
        abstract val konanDataDir: Property<String>

        /** The CMake build tree. */
        @get:Internal
        abstract val workDir: DirectoryProperty

        /** Every `.a` file the build produced, in one flat directory. */
        @get:OutputDirectory
        abstract val libraryDir: DirectoryProperty

        @TaskAction
        fun build() {
            val target = KonanTarget.predefinedTargets.getValue(konanTarget.get())
            val distribution = Distribution(konanHome.get(), konanDataDir = konanDataDir.orNull)
            val platform = PlatformManager(distribution).platform(target)
            val work = workDir.get().asFile.apply { mkdirs() }

            val toolchain =
                when (target.family) {
                    Family.OSX -> appleOptions(platform)
                    Family.LINUX -> listOf("-DCMAKE_TOOLCHAIN_FILE=${linuxToolchainFile(platform, work)}")
                    else -> throw GradleException("No CMake toolchain for ${target.name}")
                }
            cmake(
                work,
                "configure",
                listOf("-S", sourceDir.get().asFile.path, "-B", work.path) + toolchain + STATIC_OPTIONS + options.get(),
            )
            cmake(
                work,
                "build",
                listOf("--build", work.path, "--parallel", Runtime.getRuntime().availableProcessors().toString()) +
                    buildTargets.get().flatMap { listOf("--target", it) },
            )

            val libraries = libraryDir.get().asFile
            libraries.deleteRecursively()
            libraries.mkdirs()
            work
                .walkTopDown()
                .filter { it.isFile && it.extension == "a" }
                .forEach { it.copyTo(File(libraries, it.name)) }
        }

        private fun appleOptions(platform: Platform): List<String> {
            val apple = platform.configurables as AppleConfigurables
            return listOf(
                "-DCMAKE_OSX_ARCHITECTURES=${apple.arch}",
                // Kotlin/Native links for this version; objects built for a newer one make ld warn.
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=${apple.osVersionMin}",
            )
        }

        private fun linuxToolchainFile(
            platform: Platform,
            work: File,
        ): File {
            val gcc = platform.configurables as GccConfigurables
            val llvmBin = File(platform.absoluteLlvmHome, "bin")
            val triple = platform.targetTriple.toString()
            // Kotlin/Native ships llvm-ar without llvm-ranlib; llvm-ar acts as ranlib under that name.
            val ranlib = File(work, "llvm-ranlib")
            if (!Files.isSymbolicLink(ranlib.toPath())) {
                Files.createSymbolicLink(ranlib.toPath(), File(llvmBin, "llvm-ar").toPath())
            }
            return File(work, "kotlin-native-toolchain.cmake").apply {
                writeText(
                    """
                    |set(CMAKE_SYSTEM_NAME Linux)
                    |set(CMAKE_SYSTEM_PROCESSOR ${platform.targetTriple.architecture})
                    |set(CMAKE_C_COMPILER "${File(llvmBin, "clang")}")
                    |set(CMAKE_CXX_COMPILER "${File(llvmBin, "clang++")}")
                    |set(CMAKE_ASM_COMPILER "${File(llvmBin, "clang")}")
                    |foreach(lang C CXX ASM)
                    |  set(CMAKE_${'$'}{lang}_COMPILER_TARGET $triple)
                    |  set(CMAKE_${'$'}{lang}_FLAGS_INIT "--gcc-toolchain=${gcc.absoluteGccToolchain}")
                    |endforeach()
                    |set(CMAKE_SYSROOT "${platform.absoluteTargetSysRoot}")
                    |set(CMAKE_AR "${File(llvmBin, "llvm-ar")}")
                    |set(CMAKE_RANLIB "$ranlib")
                    |# Compiler checks must not link: the sysroot is for another machine.
                    |set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
                    |set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
                    |set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
                    |set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
                    |set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
                    |
                    """.trimMargin(),
                )
            }
        }

        /** Runs CMake with its output in a log file, which is only shown when the step fails. */
        private fun cmake(
            work: File,
            step: String,
            arguments: List<String>,
        ) {
            val log = File(work, "cmake-$step.log")
            val result =
                log.outputStream().use { output ->
                    exec.exec {
                        commandLine(listOf("cmake") + arguments)
                        standardOutput = output
                        errorOutput = output
                        isIgnoreExitValue = true
                    }
                }
            if (result.exitValue != 0) {
                val tail = log.readLines().takeLast(LOG_TAIL_LINES).joinToString("\n")
                throw GradleException("cmake $step failed for ${konanTarget.get()} (full log: $log):\n$tail")
            }
        }

        private companion object {
            val STATIC_OPTIONS =
                listOf(
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=OFF",
                    // Kotlin/Native links position-independent executables.
                    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                )
            const val LOG_TAIL_LINES = 40
        }
    }
