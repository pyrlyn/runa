plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(projects.platform)
            implementation(libs.okio)
            // Main code here is test code elsewhere.
            implementation(kotlin("test"))
        }
    }
}
