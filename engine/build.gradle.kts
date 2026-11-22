plugins {
    id("runa.library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(projects.core)
            api(libs.okio)
            api(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
            implementation(projects.engineTesting)
        }
    }
}
