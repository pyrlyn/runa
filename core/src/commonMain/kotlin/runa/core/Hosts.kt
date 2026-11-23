package runa.core

/**
 * True when [host] can only mean this machine, so that what listens there cannot be reached from
 * another one. A name that merely starts like a loopback address (`127.example.com`) is not.
 */
public fun isLoopbackHost(host: String): Boolean = host.lowercase() in LOOPBACK_NAMES || LOOPBACK_V4.matches(host)

private val LOOPBACK_NAMES = setOf("localhost", "::1")
private val LOOPBACK_V4 = Regex("127(\\.\\d{1,3}){3}")

/** The base URL of an HTTP server on [host]; an IPv6 address needs brackets there. */
public fun httpUrl(
    host: String,
    port: Int,
): String = "http://${if (':' in host) "[$host]" else host}:$port"
