plugins {
    id("runa.library")
    id("runa.serialization")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.gguf)
            api(projects.config)
            implementation(libs.kotlinx.serialization.json)
        }
    }
}
