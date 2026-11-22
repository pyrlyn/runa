plugins {
    id("runa.library")
    id("runa.serialization")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            // The daemon API types are @Serializable, so their serializers are part of this module's API.
            api(libs.kotlinx.serialization.json)
        }
    }
}
