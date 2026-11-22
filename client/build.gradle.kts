plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.core)
            api(libs.ktor.client.core)
        }
    }
}
