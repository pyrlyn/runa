plugins {
    id("runa.library")
    id("runa.serialization")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(projects.client)
            implementation(projects.store)
            implementation(projects.hub)
            api(projects.config)
            implementation(libs.clikt)
            implementation(libs.mordant)
            implementation(libs.kotlinx.serialization.json)
            implementation(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(libs.ktor.client.mock)
        }
    }
}
