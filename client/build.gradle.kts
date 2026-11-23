plugins {
    id("runa.library")
    id("runa.serialization")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.core)
            api(libs.ktor.client.core)
            // Named here although :core exports it: the metadata compilation does not see it through :core.
            implementation(libs.kotlinx.serialization.json)
        }
        commonTest.dependencies {
            implementation(libs.ktor.client.mock)
            implementation(projects.testing)
        }
    }
}
