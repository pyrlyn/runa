plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.engine)
        }
    }
}
