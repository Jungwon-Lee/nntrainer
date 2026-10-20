---
title: Building with the Hexagon HTP Backend
...

# Building with the Hexagon HTP Backend

This guide builds the native Hexagon HTP runtime for Android ARM64, then
optionally packages it as a QuickDotAI AAR and SampleTestAPP APK. The same
build serves the supported Gemma4 and LFM2 model packs. Model weights are
prepared and deployed separately.

For supported devices, model formats and performance, see the
[Hexagon HTP overview](../hexagon-htp.md). The native HTP build uses the
Hexagon SDK and HexKL. The separate QNN build is covered in
[QNN build instructions](QNN_BUILD.md).

## Prerequisites

### Host and toolchain

- **Host used for validation:** Ubuntu 24.04 on x86_64. The HTP scripts use
  the SDK's Ubuntu `qaic` executable.
- **Build tools:** Bash, Git, Python 3, Meson, Ninja, a host C/C++ toolchain,
  and the repository's Android build dependencies. See
  [Getting Started](../getting-started.md) for general build setup.
- **Tokenizer build:** Rust and Cargo, with the `aarch64-linux-android`
  target. The tokenizer helper can install the target through `rustup`.
- **Android NDK:** r30 in the validated setup.
- **Hexagon SDK:** 6.4.0.1, including FastRPC tools and headers.
- **Hexagon Tools:** 19.0.04.
- **HexKL:** 1.0 beta 2, with both the `hexagon_toolv19_v81` and
  `armv8_android26` libraries under `lib/6.4.0.2`.
- **Dependency access:** Initial builds may download Git submodules,
  generated-header dependencies and Rust crates. App builds also resolve
  Gradle dependencies.

Obtain the Hexagon SDK and HexKL separately from Qualcomm. They are not
included in the repository. The Hexagon SDK version (`6.4.0.1`) and HexKL
library directory (`6.4.0.2`) are independent settings.

### Additional requirements for Android packages

- **Java:** JDK 17.
- **Android SDK:** platform 36, build-tools 36.0.0 and platform-tools;
  build-tools 35.0.0 was also used by Gradle in the documented setup.
- **CMake:** Android SDK CMake 3.22.1.
- **Package ABI:** `arm64-v8a`.
- **SDK levels:** QuickDotAI and SampleTestAPP use `compileSdk=36` and
  `minSdk=33`; SampleTestAPP targets SDK 36. These package settings do not
  establish HTP support on every device at those Android API levels.

### Device requirements for deployment

- **Standard DSP target:** Hexagon v81; the build script requires
  `HEX_ARCH=v81`.
- **Validated devices:** Galaxy S26 Ultra SM-S948N and SM-S948U, with
  Android 16 and 4 KB pages in the app validation setup.
- **Connection:** USB debugging enabled and an authorized `adb` connection.
- **Device selection:** Set `ANDROID_SERIAL` when multiple devices are
  connected.

The current package has not been validated on 16 KB-page devices. HexKL's
`libsdkl.so` has 4 KB LOAD alignment, and the DSP ELF is packaged in the
native library directory. See the
[packaging limits](../gemma4-htp-quickdotai.md#known-limits).

## Configure the environment

Start from a source checkout containing the native HTP backend. Run the
following commands from the repository root. Replace `/path/to/...` with
absolute paths to your installations.

```bash
git submodule sync
git submodule update --init --recursive --depth 1

export ANDROID_NDK=/path/to/android-ndk-r30
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.1
source "$HEXAGON_SDK_ROOT/setup_sdk_env.source"
export DEFAULT_HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"
export HEXKL_ROOT=/path/to/hexkl_addon
export HEXKL_SDK_VER=6.4.0.2
export HEXKL_TOOLS_VARIANT=toolv19
export HEXKL_ARCH=armv8_android26
export HEX_ARCH=v81
```

The SDK setup script initializes its environment. Set
`DEFAULT_HEXAGON_TOOLS_ROOT` to the installed toolchain root containing
`Tools/bin/hexagon-clang`, adjusting the example if your installation uses
a different layout. Confirm that the selected tools and libraries exist:

```bash
test -x "$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
test -x "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang"
test -f "$HEXKL_ROOT/lib/$HEXKL_SDK_VER/hexagon_${HEXKL_TOOLS_VARIANT}_${HEX_ARCH}/libhexkl_micro.a"
test -f "$HEXKL_ROOT/lib/$HEXKL_SDK_VER/$HEXKL_ARCH/libsdkl.so"
```

Each check should return exit status zero. For AAR/APK builds, also set:

```bash
export JAVA_HOME=/path/to/jdk-17
export ANDROID_HOME=/path/to/android-sdk
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"
```

## Build the native runtime

Use the shared Android build entry point:

```bash
bash Applications/CausalLM/build_android.sh --htp
```

The script performs these steps:

1. Generates the ARM FastRPC stub from `test/htp/nntr_hvx.idl`.
2. Generates the DSP skeleton and builds `libnntr_hvx_skel.so` for v81.
3. Prepares missing source dependencies and the Android tokenizer archive.
4. Builds the nntrainer engine with HTP enabled and QNN disabled.
5. Builds the CausalLM CLI, QuickDotAI C API and native tools against that
   engine's ABI configuration.

No connected device is needed for this command. The default HTP compute
thread count is eight; use `--nntr-threads=N` to select another positive
count. `--htp` and `--qnn` cannot be combined.

### Native artifacts

All paths below are relative to the repository root.

| Location | Artifacts and purpose |
|---|---|
| `builddir_app/htp/` | `nntr_causallm`, the model CLI; `quick_dot_ai_test`, the C API test executable; `nntr_quantize` and `nntr_safetensors_info`, native tools |
| `builddir_app/htp/` | `libcausallm.so` and `libquick_dot_ai_api.so`, the model implementation and public C API |
| `builddir/android_build_result/lib/arm64-v8a/` | `libnntrainer.so` and `libccapi-nntrainer.so`, the engine libraries |
| `test/htp/build/` | `libnntr_hvx_skel.so`, the DSP library |
| `nntrainer/tensor/htp_backend/generated/` | `nntr_hvx.h` and `nntr_hvx_stub.c`, generated ARM interface sources compiled into the backend |

The runtime also needs `libc++_shared.so` from the selected NDK and
`libsdkl.so` from the selected HexKL ARM library directory. The build script
collects them for packaging or deployment.

## Build the AAR and sample APK

```bash
bash Applications/CausalLM/build_android.sh --htp --app
```

This builds the native runtime, stages its libraries in
`Applications/CausalLM/Android/QuickDotAI/prebuilt_libs/`, and runs
`:QuickDotAI:assembleDebug` and `:SampleTestAPP:assembleDebug` through the
Gradle wrapper. The script passes the same NDK path to the native and JNI
builds.

| Package | Output |
|---|---|
| QuickDotAI AAR | `Applications/CausalLM/Android/QuickDotAI/build/outputs/aar/QuickDotAI-debug.aar` |
| SampleTestAPP APK | `Applications/CausalLM/Android/SampleTestAPP/build/outputs/apk/debug/SampleTestAPP-debug.apk` |
| Native package verification report | The APK path above with `.native-manifest.json` appended |

The build automatically runs
[`verify_native_package.py`](../../Applications/CausalLM/Android/verify_native_package.py)
on both packages. It checks that staged libraries are present, ARM shared
library dependencies are available in the package or recognized as system
libraries, and the DSP skeleton is a byte-identical v81 Hexagon image.
It also rejects QNN libraries in the HTP stage and a packaged SDK
`libcdsprpc.so` link stub. The FastRPC runtime comes from the device's
vendor namespace.

Package verification checks the files produced by the build. Successful
inference still requires a compatible device and model pack.

### Packaging in a consuming app

Preserve the settings used by the sample app when consuming the AAR:

```kotlin
android {
    packaging {
        jniLibs.useLegacyPackaging = true
        jniLibs.keepDebugSymbols += "**/libnntr_hvx_skel.so"
    }
}
```

The DSP skeleton must be extracted to the native library directory and
remain unchanged by Android's native stripping tools. Pass
`applicationInfo.nativeLibraryDir` as `LoadModelRequest.nativeLibDir` when
loading a native HTP model. See the
[QuickDotAI AAR instructions](../../Applications/CausalLM/Android/QuickDotAI/README.md)
for the Kotlin API.

Direct Gradle invocations require
`-PnntrainerNdkPath=<absolute NDK path>`. They consume the staged native
libraries; use the shared build script first to refresh the HTP runtime.

## Deploy to a device

Select an authorized device, then choose either native-only deployment or
deployment with the sample app:

```bash
adb devices
export ANDROID_SERIAL=your-device-serial

# Build and push the native runtime and tools.
bash Applications/CausalLM/build_android.sh --htp --install

# Alternatively, also build and install SampleTestAPP.
bash Applications/CausalLM/build_android.sh --htp --app --install
```

`ANDROID_SERIAL` is optional when exactly one authorized device is
connected. Both commands refresh the managed native artifacts in
`/data/local/tmp/Quick.AI/`. The second also installs the APK with
`adb install -r`.

The deployment creates two launch wrappers in that directory:

- `run_causallm.sh` invokes `nntr_causallm` and forwards its arguments.
- `run_test.sh` invokes `quick_dot_ai_test` and forwards its arguments.

Both wrappers set `LD_LIBRARY_PATH` for the ARM libraries,
`ADSP_LIBRARY_PATH` for the DSP skeleton, and `NNTR_NUM_THREADS` to the
selected build count. They also change the working directory to
`/data/local/tmp/Quick.AI/`.

These commands deploy binaries, not model packs. Prepare the weights,
configuration and tokenizer separately, then follow the model's launch or
app provisioning instructions:

- [Gemma4 CLI](../gemma4-htp-user-guide.md#7-run-from-the-command-line)
  and [Gemma4 app setup](../gemma4-htp-quickdotai.md).
- [LFM2 CLI](../lfm2-htp-user-guide.md#build-and-run)
  and [LFM2 API and app setup](../../Applications/CausalLM/api/README.md#lfm2-moe-on-native-htp).

## Rebuild and keep artifacts consistent

The ARM FastRPC stub and DSP skeleton implement the same interface. After
changing the IDL, shared structures, backend or DSP kernels, rebuild and
deploy the corresponding ARM libraries and DSP skeleton together. For app
use, regenerate and reinstall the package as well.

The normal `--htp` command regenerates both sides of the FastRPC interface
and rebuilds the engine. Use it after updating the source checkout.

| Option | Behavior |
|---|---|
| `--cache` | Reuses a compatible engine build and rebuilds it on a cache miss. DSP generation/build and CausalLM configuration/build still run. |
| `--clean` | Recreates the selected CausalLM build directory, such as `builddir_app/htp/`. It does not clear every generated file or dependency archive. |
| `--nntr-threads=N` | Selects the compute thread count, including the value in deployment wrappers. |

The engine cache checks build settings and required artifacts, not the
source revision. Omit `--cache` after engine or shared-interface changes.
Explicit engine options beginning with `-D` also bypass engine reuse.
The tokenizer archive is built only when missing; if tokenizer sources or
its toolchain change, run
`bash Applications/CausalLM/build_tokenizer_android.sh` to refresh it before
the main build.

For low-level development, the generation scripts are
[`generate_stub.sh`](../../nntrainer/tensor/htp_backend/generate_stub.sh)
and [`test/htp/build.sh`](../../test/htp/build.sh). Export `HEX_ARCH=v81`
before invoking the DSP script directly: its standalone default is v79.
The shared Android entry point enforces v81.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing `qaic` or `hexagon-clang` | Source the SDK environment and check the executable paths shown above. `DEFAULT_HEXAGON_TOOLS_ROOT` must contain `Tools/bin/hexagon-clang`. |
| Missing `libhexkl_micro.a` or `libsdkl.so` | Check `HEXKL_ROOT`, `HEXKL_SDK_VER`, and the DSP/ARM subdirectories. Installing SDK 6.4.0.1 does not imply a HexKL `6.4.0.1` library directory. |
| DSP target rejected | Set `HEX_ARCH=v81`. Other Hexagon targets require separate porting and validation. |
| Gradle rejects the NDK path | Use an absolute NDK root containing `source.properties`; pass it through `-PnntrainerNdkPath` for direct Gradle calls. |
| Package verification reports a changed DSP image | Preserve `libnntr_hvx_skel.so` with `jniLibs.keepDebugSymbols`, then rebuild the package. |
| Package verification reports missing or mixed libraries | Re-run `build_android.sh --htp --app` to refresh the native stage and packages from the same build. |
| ARM or DSP library fails to load on the device | Confirm that the matching runtime libraries were deployed together. Use the generated CLI wrappers; in an app, pass the extracted native library directory to the model loader. |
| Deployment cannot select a device | Check `adb devices` for an authorized device and set `ANDROID_SERIAL`. |

After building, prepare a pack with
[HTP Model Preparation](HTP_MODEL_PREPARATION.md), run it with
[HTP Inference](HTP_INFERENCE.md), and check correctness and lifecycle
behavior with [HTP Validation](HTP_VALIDATION.md).
