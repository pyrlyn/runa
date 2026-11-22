plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(projects.client)
            implementation(projects.store)
            implementation(projects.hub)
            implementation(projects.config)
            implementation(libs.clikt)
            implementation(libs.mordant)
        }
    }
}
