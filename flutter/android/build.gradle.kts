group = "com.odometersdk"
version = "1.0"

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.odometersdk"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
    implementation("org.opencv:opencv:4.9.0")
    implementation("androidx.core:core-ktx:1.13.1")
}
