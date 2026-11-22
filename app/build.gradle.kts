plugins {
    id("runa.kmp")
}

val gitSha: Provider<String> =
    providers
        .exec {
            commandLine("git", "rev-parse", "--short=12", "HEAD")
            isIgnoreExitValue = true
        }.standardOutput.asText
        .map { it.trim().ifEmpty { "unknown" } }

val buildInfoDir = layout.buildDirectory.dir("generated/buildInfo/kotlin")

val generateBuildInfo =
    tasks.register("generateBuildInfo") {
        // Locals, not script properties: the configuration cache cannot serialize the script object.
        val version = project.version.toString()
        val sha = gitSha
        val outDir = buildInfoDir
        inputs.property("version", version)
        inputs.property("gitSha", sha)
        outputs.dir(outDir)
        doLast {
            val file = outDir.get().file("runa/app/BuildInfo.kt").asFile
            file.parentFile.mkdirs()
            file.writeText(
                """
            |package runa.app
            |
            |internal object BuildInfo {
            |    const val VERSION: String = "$version"
            |    const val GIT_SHA: String = "${sha.get()}"
            |}
            |
                """.trimMargin(),
            )
        }
    }

kotlin {
    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget>().configureEach {
        binaries.executable {
            baseName = "runa"
            entryPoint = "runa.app.main"
        }
    }

    sourceSets {
        commonMain {
            kotlin.srcDir(generateBuildInfo)
            dependencies {
                implementation(projects.cli)
                implementation(projects.daemon)
                implementation(projects.engineLlama)
            }
        }
    }
}
