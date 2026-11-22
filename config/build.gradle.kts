plugins {
    id("runa.library")
    id("runa.serialization")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.core)
            api(projects.platform)
            implementation(libs.kotlinx.serialization.json)
        }
    }
}
