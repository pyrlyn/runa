pluginManagement {
    includeBuild("build-logic")
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode = RepositoriesMode.FAIL_ON_PROJECT_REPOS
    repositories {
        mavenCentral()
    }
}

enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")

rootProject.name = "runa"

// The module list and responsibilities are in PLAN.md, section 5.2.
include(
    ":core",
    ":platform",
    ":config",
    ":gguf",
    ":store",
    ":hub",
    ":engine",
    ":engine-llama",
    ":daemon",
    ":client",
    ":cli",
    ":app",
)
