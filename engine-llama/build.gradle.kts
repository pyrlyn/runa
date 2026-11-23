import org.jetbrains.kotlin.gradle.plugin.getKotlinPluginVersion
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeTest
import org.jetbrains.kotlin.konan.target.Architecture
import org.jetbrains.kotlin.konan.target.Family
import org.jetbrains.kotlin.konan.target.HostManager
import runa.gradle.CmakeStaticLibraries

plugins {
    id("runa.library")
}

val llamaDir = rootProject.layout.projectDirectory.dir("third_party/llama.cpp")

// The submodule is pinned to a commit, so the commit stands for its content in up-to-date checks.
val llamaRevision: Provider<String> =
    providers
        .exec {
            commandLine("git", "-C", llamaDir.asFile.path, "rev-parse", "HEAD")
            isIgnoreExitValue = true
        }.standardOutput.asText
        .map { it.trim().ifEmpty { "unknown" } }

// Where the Kotlin Gradle plugin unpacks Kotlin/Native; it has no public API for this path.
val nativeDataDir: Provider<String> = providers.environmentVariable("KONAN_DATA_DIR")
val nativeHome: Provider<String> =
    run {
        // Locals, not script properties: the configuration cache cannot serialize the script object.
        val distribution = "kotlin-native-prebuilt-${HostManager.platformName()}-${getKotlinPluginVersion()}"
        val userHome = System.getProperty("user.home")
        providers
            .gradleProperty("kotlin.native.home")
            .orElse(nativeDataDir.orElse("$userHome/.konan").map { "$it/$distribution" })
    }

val llamaOptions =
    listOf(
        // Only the library: runa has its own CLI, server and HTTP client.
        "-DLLAMA_BUILD_COMMON=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_OPENSSL=OFF",
        // The binary is copied to other machines, so it must not be tuned for the build host.
        "-DGGML_NATIVE=OFF",
        "-DGGML_OPENMP=OFF",
    )

fun llamaOptionsFor(target: org.jetbrains.kotlin.konan.target.KonanTarget): List<String> =
    when {
        // The shader source is compiled into the library; otherwise default.metallib must ship beside runa.
        target.family == Family.OSX -> {
            listOf("-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON")
        }

        // Cross-compiling turns ggml's x86 extensions off; without AVX2 CPU inference is several times slower.
        target.architecture == Architecture.X64 -> {
            listOf("SSE42", "AVX", "AVX2", "BMI2", "FMA", "F16C").map { "-DGGML_$it=ON" }
        }

        else -> {
            emptyList()
        }
    }

fun llamaLibrariesFor(target: org.jetbrains.kotlin.konan.target.KonanTarget): List<String> =
    listOf("libllama.a", "libggml.a", "libggml-cpu.a") +
        (if (target.family == Family.OSX) listOf("libggml-metal.a", "libggml-blas.a") else emptyList()) +
        "libggml-base.a"

// llama.cpp states its version only in CMake; EngineInfo reports it.
val llamaVersion: Provider<String> =
    providers.fileContents(llamaDir.file("CMakeLists.txt")).asText.map { cmake ->
        listOf("MAJOR", "MINOR", "PATCH").joinToString(".") { part ->
            Regex("""set\(LLAMA_VERSION_$part (\d+)\)""").find(cmake)?.groupValues?.get(1) ?: "0"
        }
    }

val llamaBuildDir = layout.buildDirectory.dir("generated/llamaBuild/kotlin")

val generateLlamaBuild =
    tasks.register("generateLlamaBuild") {
        // Locals, not script properties: the configuration cache cannot serialize the script object.
        val version = llamaVersion
        val outDir = llamaBuildDir
        inputs.property("version", version)
        outputs.dir(outDir)
        doLast {
            val file = outDir.get().file("runa/engine/llama/LlamaBuild.kt").asFile
            file.parentFile.mkdirs()
            file.writeText(
                """
                |package runa.engine.llama
                |
                |internal object LlamaBuild {
                |    const val VERSION: String = "${version.get()}"
                |}
                |
                """.trimMargin(),
            )
        }
    }

kotlin {
    targets.withType<KotlinNativeTarget>().configureEach {
        val konan = konanTarget
        val interop =
            compilations.getByName("main").cinterops.create("llama") {
                includeDirs(llamaDir.dir("include"), llamaDir.dir("ggml/include"))
            }
        // llama.cpp is not built for Windows yet (PLAN.md, section 13): mingwX64 gets the bindings
        // and compiles, but cannot link.
        if (konan.family == Family.MINGW || !HostManager().isEnabled(konan)) return@configureEach

        val targetName = name.replaceFirstChar(Char::uppercase)
        val libraryDir = layout.buildDirectory.dir("llama/${konan.name}/lib")
        val cmake =
            tasks.register<CmakeStaticLibraries>("cmakeLlama$targetName") {
                description = "Builds llama.cpp's static libraries for ${konan.name}."
                sourceDir.set(llamaDir)
                sourceRevision.set(llamaRevision)
                konanTarget.set(konan.name)
                options.set(llamaOptions + llamaOptionsFor(konan))
                buildTargets.set(listOf("llama"))
                konanHome.set(nativeHome)
                konanDataDir.set(nativeDataDir)
                workDir.set(layout.buildDirectory.dir("llama/${konan.name}/cmake"))
                this.libraryDir.set(libraryDir)
                // Kotlin/Native downloads its compiler, clang and sysroots when its first task for
                // a target runs. CMake compiles with them, so it waits for such a task.
                dependsOn(":engine:compileKotlin$targetName")
            }
        interop.extraOpts(
            listOf("-libraryPath", libraryDir.get().asFile.path) +
                llamaLibrariesFor(konan).flatMap { listOf("-staticLibrary", it) },
        )
        tasks.named(interop.interopProcessingTaskName) { inputs.files(cmake) }
    }

    sourceSets {
        all {
            languageSettings.optIn("kotlinx.cinterop.ExperimentalForeignApi")
        }
        commonMain {
            kotlin.srcDir(generateLlamaBuild)
            dependencies {
                api(projects.engine)
                implementation(projects.gguf)
                implementation(projects.platform)
            }
        }
        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}

// Tests that need a real model read its path from this variable; without it they skip, so the
// result of a run depends on it.
tasks.withType<KotlinNativeTest>().configureEach {
    inputs.property("runa.testModel", providers.environmentVariable("RUNA_TEST_MODEL").orElse(""))
}
