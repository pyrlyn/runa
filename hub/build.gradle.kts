plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(libs.ktor.client.core)
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
