# cucorr

## Building cucorr

cucorr requires LLVM version 12 and CMake version >= 3.20. Use CMake presets `develop` or `release`
to build.

### 2.1 Build example

cucorr uses CMake to build. Example build recipe (release build, installs to default prefix
`${cucorr_SOURCE_DIR}/install/cucorr`)

```sh
$> cd cucorr
$> cmake --preset release
$> cmake --build build --target install --parallel
```
