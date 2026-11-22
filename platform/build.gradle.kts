plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(libs.okio)
        }
    }
}
