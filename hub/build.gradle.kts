plugins {
    id("runa.library")
    id("runa.serialization")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.core)
            api(projects.gguf)
            api(libs.ktor.client.core)
            implementation(libs.kotlinx.serialization.json)
        }
        commonTest.dependencies {
            implementation(libs.ktor.client.mock)
        }
        // The CIO client has no TLS on native, so HTTPS to Hugging Face needs each OS's own engine.
        macosMain.dependencies {
            implementation(libs.ktor.client.darwin)
        }
        linuxMain.dependencies {
            implementation(libs.ktor.client.curl)
        }
        findByName("mingwMain")?.dependencies {
            implementation(libs.ktor.client.winhttp)
        }
    }
}
