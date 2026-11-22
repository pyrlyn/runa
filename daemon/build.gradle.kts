plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(projects.engine)
            implementation(projects.store)
            implementation(projects.hub)
            implementation(libs.ktor.server.cio)
        }
        commonTest.dependencies {
            // Plain HTTP on loopback, so the CIO client is enough here.
            implementation(libs.ktor.client.cio)
        }
    }
}
