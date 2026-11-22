plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.gguf)
            api(projects.config)
        }
    }
}
