# Contributing

runa is being rewritten as a Kotlin/Native CLI. Read [`PLAN.md`](PLAN.md) first: it lists the
phases, their acceptance criteria and the checks each pull request must pass.

- Repository prose is English.
- Every change goes through a pull request; CI must be green before merge.
- Toolchain: `mise install` (JDK, CMake, lint tools), then `./gradlew build`. `./gradlew ktlintFormat`
  fixes formatting; library modules use `explicitApi()`.
