// Kotlin Multiplatform with runa's native targets, ktlint and kotlin.test; applied by every module.
plugins {
    id("org.jetbrains.kotlin.multiplatform")
    id("org.jlleitschuh.gradle.ktlint")
}

kotlin {
    macosArm64()
    linuxX64()
    linuxArm64()
    if (providers.gradleProperty("runa.mingw").orNull == "true") {
        mingwX64()
    }

    // unixMain/unixTest sit between native and linux/apple for POSIX-only code (fork, signals,
    // posix_spawn) that mingwX64 cannot share.
    @OptIn(org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi::class)
    applyDefaultHierarchyTemplate {
        common {
            group("native") {
                group("unix") {
                    group("linux")
                    group("apple")
                }
            }
        }
    }

    compilerOptions {
        allWarningsAsErrors.set(true)
    }

    sourceSets {
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
    }
}

ktlint {
    version.set(
        extensions
            .getByType<VersionCatalogsExtension>()
            .named("libs")
            .findVersion("ktlint")
            .get()
            .requiredVersion,
    )
    filter {
        // Generated sources (BuildInfo.kt) are not ours to format.
        exclude { it.file.path.contains("/build/generated/") }
    }
}
