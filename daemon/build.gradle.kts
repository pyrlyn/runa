plugins {
    id("runa.library")
    id("runa.serialization")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.engine)
            api(projects.config)
            implementation(projects.store)
            implementation(libs.ktor.server.cio)
            implementation(libs.kotlinx.serialization.json)
        }
        commonTest.dependencies {
            implementation(projects.client)
            implementation(projects.engineTesting)
            implementation(projects.gguf)
            implementation(libs.kotlinx.coroutines.test)
            // Plain HTTP on loopback, so the CIO client is enough here.
            implementation(libs.ktor.client.cio)
        }
    }
}
