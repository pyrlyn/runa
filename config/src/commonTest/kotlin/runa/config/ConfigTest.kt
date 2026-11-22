package runa.config

import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath
import runa.core.UserInputError
import runa.platform.RunaPaths
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class ConfigTest {
    private val fs = FileSystem.SYSTEM
    private val dir: Path = FileSystem.SYSTEM_TEMPORARY_DIRECTORY / "runa-config-test-${Random.nextLong().toULong()}"
    private val paths = RunaPaths(dir, dir, dir)

    @AfterTest
    fun cleanUp() = fs.deleteRecursively(dir)

    private fun load() = ConfigFile.load(fs, paths.configFile)

    private fun resolver(
        flags: Map<ConfigKey, String> = emptyMap(),
        env: Map<String, String> = emptyMap(),
    ) = ConfigResolver(flags, env::get, ::load, paths)

    @Test
    fun precedenceIsFlagThenEnvThenFileThenDefault() {
        val key = ConfigKey.HOST
        assertEquals(ResolvedSetting(key, "127.0.0.1", ConfigSource.DEFAULT), resolver().resolve(key))

        load().with(key, "from-file").save(fs)
        assertEquals(ResolvedSetting(key, "from-file", ConfigSource.FILE), resolver().resolve(key))

        val env = mapOf("RUNA_HOST" to "from-env")
        assertEquals(ResolvedSetting(key, "from-env", ConfigSource.ENV), resolver(env = env).resolve(key))

        val flags = mapOf(key to "from-flag")
        assertEquals(ResolvedSetting(key, "from-flag", ConfigSource.FLAG), resolver(flags, env).resolve(key))
    }

    @Test
    fun everyEnvironmentVariableOverridesItsKey() {
        val env = mapOf("RUNA_HOST" to "h", "RUNA_PORT" to "4321", "RUNA_BACKEND" to "lmstudio", "RUNA_LOG_LEVEL" to "debug")
        val resolver = resolver(env = env)
        assertEquals("h", resolver.resolve(ConfigKey.HOST).value)
        assertEquals("4321", resolver.resolve(ConfigKey.PORT).value)
        assertEquals("lmstudio", resolver.resolve(ConfigKey.BACKEND).value)
        assertEquals("debug", resolver.resolve(ConfigKey.LOG_LEVEL).value)
    }

    @Test
    fun unsetKeysAndTheModelsDirDefault() {
        assertEquals(ResolvedSetting(ConfigKey.PORT, null, ConfigSource.DEFAULT), resolver().resolve(ConfigKey.PORT))
        assertEquals((dir / "models").toString(), resolver().resolve(ConfigKey.MODELS_DIR).value)
        // An empty variable counts as unset, like in a shell.
        assertEquals(ConfigSource.DEFAULT, resolver(env = mapOf("RUNA_HOST" to "")).resolve(ConfigKey.HOST).source)
    }

    @Test
    fun invalidValuesNameTheirOrigin() {
        val fromEnv = assertFailsWith<UserInputError> { resolver(env = mapOf("RUNA_PORT" to "http")).resolve(ConfigKey.PORT) }
        assertEquals("RUNA_PORT must be a port number from 1 to 65535", fromEnv.message)
        val fromFlag =
            assertFailsWith<UserInputError> { resolver(flags = mapOf(ConfigKey.LOG_LEVEL to "loud")).resolve(ConfigKey.LOG_LEVEL) }
        assertEquals("--log-level must be one of: debug, info, warn, error", fromFlag.message)
        assertFailsWith<UserInputError> { resolver(env = mapOf("RUNA_BACKEND" to "ollama")).checkOverrides() }
        assertFailsWith<UserInputError> { resolver(flags = mapOf(ConfigKey.HOST to "a b")).checkOverrides() }
        resolver(flags = mapOf(ConfigKey.HOST to "example.org"), env = mapOf("RUNA_PORT" to "1")).checkOverrides()
        assertFailsWith<UserInputError> { load().with(ConfigKey.GPU, "1.5") }
        assertFailsWith<UserInputError> { load().with(ConfigKey.TTL, "-1") }
        assertFailsWith<UserInputError> { load().with(ConfigKey.SERVER_CORS, "yes") }
    }

    @Test
    fun savesTypedNestedJsonAndKeepsUnknownKeys() {
        fs.createDirectories(dir)
        fs.write(paths.configFile) { writeUtf8("""{"future_key": [1, 2], "server": {"tls": true}}""") }
        load()
            .with(ConfigKey.SERVER_PORT, "8080")
            .with(ConfigKey.SERVER_CORS, "true")
            .with(ConfigKey.GPU, "0.5")
            .save(fs)

        val text = fs.read(paths.configFile) { readUtf8() }
        assertContains(text, "\"future_key\"")
        assertContains(text, "\"tls\": true")
        assertContains(text, "\"port\": 8080")
        assertContains(text, "\"cors\": true")
        assertContains(text, "\"gpu\": \"0.5\"")
        assertEquals("8080", load()[ConfigKey.SERVER_PORT])
        assertNull(load()[ConfigKey.TTL])
    }

    @Test
    fun badFilesAreUserErrors() {
        fs.createDirectories(dir)
        fs.write(paths.configFile) { writeUtf8("{ not json") }
        assertContains(assertFailsWith<UserInputError> { load() }.message.orEmpty(), "is not valid JSON")

        fs.write(paths.configFile) { writeUtf8("[]") }
        assertContains(assertFailsWith<UserInputError> { load() }.message.orEmpty(), "must contain a JSON object")

        fs.write(paths.configFile) { writeUtf8("""{"port": 70000, "server": {"cors": {}}}""") }
        assertContains(assertFailsWith<UserInputError> { load()[ConfigKey.PORT] }.message.orEmpty(), "port must be a port number")
        assertContains(assertFailsWith<UserInputError> { load()[ConfigKey.SERVER_CORS] }.message.orEmpty(), "server.cors must be a bool")
    }

    @Test
    fun aFileWhereTheDirectoryShouldBeIsAUserError() {
        fs.write(dir) { writeUtf8("not a directory") }
        assertContains(assertFailsWith<UserInputError> { load() }.message.orEmpty(), "Cannot read")
        assertContains(assertFailsWith<UserInputError> { ConfigFile.load(fs, dir / "sub" / "config.json") }.message.orEmpty(), "Cannot")
    }

    @Test
    fun keysParseByIdAndNameTheirFlag() {
        assertEquals(ConfigKey.SERVER_PORT, ConfigKey.parse("server.port"))
        assertNull(ConfigKey.parse("nope"))
        assertEquals("--log-level", ConfigKey.LOG_LEVEL.flag)
        assertEquals("/a/b".toPath(), RunaPaths("/a".toPath(), "/a".toPath(), "/a".toPath()).configDir / "b")
    }
}
