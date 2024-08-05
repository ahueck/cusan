# cusan

## Building cusan

cusan requires LLVM version 12 and CMake version >= 3.20. Use CMake presets `develop` or `release`
to build.

### 2.1 Build example

cusan uses CMake to build. Example build recipe (release build, installs to default prefix
`${cusan_SOURCE_DIR}/install/cusan`)

```sh
$> cd cusan
$> cmake --preset release
$> cmake --build build --target install --parallel
```
