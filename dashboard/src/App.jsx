import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

const empty = {
  token_speed: { toks_per_s: 0, decode_toks_per_s: 0, history: [] },
  in_flight: [],
  system_load: { process_cpu_percent: 0, process_rss_mib: 0 },
  latency: { p50_ms: 0, p95_ms: 0, p99_ms: 0, samples: 0 },
  errors: { count: 0, recent: [] },
  quotas: {
    parallel_limit: 0,
    parallel_in_use: 0,
    tokens_per_minute: 0,
    requests_per_second: 0,
    requests_in_window: 0,
    window_secs: 60,
    default_tokens_per_minute: 100000,
    default_requests_per_second: 16,
    hitting_ceiling: false,
  },
  daemon: { status: "not_this_process", alive: null, detail: "" },
  models: { configured: 0, loaded_count: 0, loaded: [] },
  mcp_tools: [],
  queue: { depth: 0, waiting: [] },
  reasoning: { tokens_spent: 0, active_budget_tokens: null },
};

function num(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "0";
  return n.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function int(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "0";
  return Math.round(n).toLocaleString();
}

function mib(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return "0 MiB";
  return `${num(n / (1024 * 1024), 1)} MiB`;
}

async function loadSnapshot() {
  const res = await fetch("/dashboard/snapshot");
  if (!res.ok) throw new Error(`snapshot ${res.status}`);
  return res.json();
}

export default function App() {
  const queryClient = useQueryClient();
  const [link, setLink] = useState("connecting");
  const query = useQuery({ queryKey: ["snapshot"], queryFn: loadSnapshot });

  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/dashboard/ws`);
    ws.onopen = () => setLink("live");
    ws.onclose = () => setLink("closed");
    ws.onerror = () => setLink("closed");
    ws.onmessage = (ev) => {
      try {
        queryClient.setQueryData(["snapshot"], JSON.parse(ev.data));
      } catch {
        /* ignore a bad frame */
      }
    };
    return () => ws.close();
  }, [queryClient]);

  const snap = query.data ?? empty;

  return (
    <main className="min-h-screen bg-stone-950 text-stone-100">
      <div className="mx-auto max-w-6xl px-4 py-6">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-widest text-amber-400">runa serve</p>
            <h1 className="text-2xl font-semibold">Live dashboard</h1>
          </div>
          <p className="text-sm text-stone-400">
            Socket{" "}
            <span className={link === "live" ? "text-emerald-400" : "text-stone-300"}>{link}</span>
            {query.isError ? (
              <span className="ml-3 text-red-400">snapshot: {query.error.message}</span>
            ) : null}
          </p>
        </header>

        {snap.quotas.hitting_ceiling ? (
          <p className="mb-4 rounded border border-amber-700 bg-amber-950 px-3 py-2 text-sm text-amber-200">
            Hitting the ceiling — parallel slots are full, or the {snap.quotas.window_secs}s window
            is over {int(snap.quotas.default_tokens_per_minute)} tokens or{" "}
            {num(snap.quotas.default_requests_per_second, 0)} requests/s.
          </p>
        ) : null}

        <section className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Tile label="Token speed" value={`${num(snap.token_speed.toks_per_s)} tok/s`} hint="generated, 60s average" />
          <Tile label="In flight" value={int(snap.in_flight.length)} hint="request / response pairs" />
          <Tile label="Queue" value={int(snap.queue.depth)} hint="waiting on the parallel limit" />
          <Tile
            label="This process"
            value={`${num(snap.system_load.process_cpu_percent)}% CPU`}
            hint={`${int(snap.system_load.process_rss_mib)} MiB RSS`}
          />
        </section>

        <section className="mb-4 grid gap-3 lg:grid-cols-2">
          <Card title="Token speed">
            <SpeedGraph history={snap.token_speed.history} />
            <p className="mt-2 text-xs text-stone-400">
              Latest engine decode rate {num(snap.token_speed.decode_toks_per_s)} tok/s. Each bar is
              generated tokens in one second.
            </p>
          </Card>
          <Card title="Latency">
            <Latency latency={snap.latency} />
          </Card>
        </section>

        <section className="mb-4">
          <Card title="Quotas and limits">
            <Quotas quotas={snap.quotas} />
          </Card>
        </section>

        <section className="mb-4 grid gap-3 lg:grid-cols-2">
          <Card title="In-flight requests">
            <InFlight rows={snap.in_flight} />
          </Card>
          <Card title="Request queue">
            <Queue queue={snap.queue} />
          </Card>
        </section>

        <section className="mb-4">
          <Card title={`Errors (${int(snap.errors.count)})`}>
            <Errors errors={snap.errors} />
          </Card>
        </section>

        <section className="grid gap-3 md:grid-cols-2">
          <Card title="Loaded models">
            <Models models={snap.models} />
          </Card>
          <Card title="Warm daemon">
            <Daemon daemon={snap.daemon} />
          </Card>
          <Card title="MCP tools">
            <Mcp tools={snap.mcp_tools} />
          </Card>
          <Card title="Reasoning budget">
            <Reasoning reasoning={snap.reasoning} />
          </Card>
        </section>
      </div>
    </main>
  );
}

function Tile({ label, value, hint }) {
  return (
    <div className="rounded border border-stone-800 bg-stone-900 px-3 py-3">
      <p className="text-xs uppercase tracking-wide text-stone-400">{label}</p>
      <p className="mt-1 text-xl font-medium">{value}</p>
      <p className="text-xs text-stone-500">{hint}</p>
    </div>
  );
}

function Card({ title, children }) {
  return (
    <section className="rounded border border-stone-800 bg-stone-900 p-3">
      <h2 className="mb-3 text-sm font-medium text-stone-200">{title}</h2>
      {children}
    </section>
  );
}

function SpeedGraph({ history }) {
  const values = Array.isArray(history) && history.length > 0 ? history.map(Number) : [0];
  const max = Math.max(1, ...values);
  const sum = values.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
  const w = 320;
  const h = 80;
  const bar = w / values.length;
  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="h-24 w-full text-amber-400" role="img" aria-label="Token speed">
        {values.map((v, i) => {
          const bh = ((Number.isFinite(v) ? v : 0) / max) * (h - 4);
          return (
            <rect
              key={i}
              x={i * bar + 0.5}
              y={h - bh}
              width={Math.max(bar - 1, 0.5)}
              height={bh}
              fill="currentColor"
            />
          );
        })}
      </svg>
      {sum === 0 ? <p className="text-sm text-stone-500">No generated tokens in this window.</p> : null}
    </div>
  );
}

function Latency({ latency }) {
  const samples = Number(latency?.samples) || 0;
  if (samples === 0) {
    return <p className="text-sm text-stone-500">No completed requests yet.</p>;
  }
  const rows = [
    ["p50", Number(latency.p50_ms) || 0],
    ["p95", Number(latency.p95_ms) || 0],
    ["p99", Number(latency.p99_ms) || 0],
  ];
  const max = Math.max(1, ...rows.map(([, v]) => v));
  return (
    <div className="space-y-3">
      {rows.map(([label, value]) => (
        <div key={label}>
          <div className="mb-1 flex justify-between text-xs text-stone-400">
            <span>{label}</span>
            <span>{num(value)} ms</span>
          </div>
          <div className="h-2 rounded bg-stone-800">
            <div className="h-2 rounded bg-amber-400" style={{ width: `${(value / max) * 100}%` }} />
          </div>
        </div>
      ))}
      <p className="text-xs text-stone-500">{int(samples)} samples in the latency ring.</p>
    </div>
  );
}

function Quotas({ quotas }) {
  const q = quotas ?? empty.quotas;
  const tokenPct = Math.min(100, (Number(q.tokens_per_minute) / Math.max(1, Number(q.default_tokens_per_minute))) * 100);
  const reqPct = Math.min(
    100,
    (Number(q.requests_per_second) / Math.max(0.001, Number(q.default_requests_per_second))) * 100,
  );
  return (
    <div className="space-y-3 text-sm">
      <Meter
        label="Tokens / minute"
        value={`${int(q.tokens_per_minute)} / ${int(q.default_tokens_per_minute)}`}
        pct={tokenPct}
      />
      <Meter
        label="Requests / second"
        value={`${num(q.requests_per_second)} / ${num(q.default_requests_per_second, 0)}`}
        pct={reqPct}
      />
      <p className="text-stone-400">
        Parallel slots {int(q.parallel_in_use)} / {int(q.parallel_limit)}. Window {int(q.window_secs)}s,{" "}
        {int(q.requests_in_window)} requests. Ceiling {q.hitting_ceiling ? "yes" : "no"}.
      </p>
    </div>
  );
}

function Meter({ label, value, pct }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-stone-400">
        <span>{label}</span>
        <span>{value}</span>
      </div>
      <div className="h-2 rounded bg-stone-800">
        <div className="h-2 rounded bg-stone-300" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function InFlight({ rows }) {
  if (!rows || rows.length === 0) {
    return <p className="text-sm text-stone-500">No requests in flight.</p>;
  }
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-xs uppercase text-stone-500">
        <tr>
          <th className="pb-1 font-medium">Route</th>
          <th className="pb-1 font-medium">Model</th>
          <th className="pb-1 text-right font-medium">Elapsed</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id} className="border-t border-stone-800">
            <td className="py-1 pr-2">{row.route}</td>
            <td className="py-1 pr-2 text-stone-300">{row.model || "—"}</td>
            <td className="py-1 text-right">{int(row.elapsed_ms)} ms</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Queue({ queue }) {
  const waiting = queue?.waiting ?? [];
  if (waiting.length === 0) {
    return <p className="text-sm text-stone-500">Queue is empty.</p>;
  }
  return (
    <ul className="space-y-1 text-sm">
      {waiting.map((row) => (
        <li key={row.id} className="flex justify-between border-t border-stone-800 py-1">
          <span>{row.route}</span>
          <span className="text-stone-400">{int(row.wait_ms)} ms</span>
        </li>
      ))}
    </ul>
  );
}

function Errors({ errors }) {
  const recent = errors?.recent ?? [];
  if (!errors || errors.count === 0) {
    return <p className="text-sm text-stone-500">No errors.</p>;
  }
  if (recent.length === 0) {
    return <p className="text-sm text-stone-400">{int(errors.count)} errors; none retained.</p>;
  }
  return (
    <ul className="space-y-2">
      {recent.map((err, i) => (
        <li key={`${err.at_unix_ms}-${i}`} className="rounded border border-stone-800 px-2 py-2 text-sm">
          <p className="text-red-300">{err.message}</p>
          <details className="mt-1">
            <summary className="cursor-pointer text-xs text-stone-500">Traceback</summary>
            <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-xs text-stone-400">{err.traceback}</pre>
          </details>
        </li>
      ))}
    </ul>
  );
}

function Models({ models }) {
  const loaded = models?.loaded ?? [];
  return (
    <div className="text-sm">
      <p className="mb-2 text-stone-400">
        {int(models?.loaded_count)} loaded, {int(models?.configured)} configured. Memory is the on-disk weight size.
      </p>
      {loaded.length === 0 ? (
        <p className="text-stone-500">No models loaded.</p>
      ) : (
        <ul className="space-y-1">
          {loaded.map((m) => (
            <li key={m.id} className="flex justify-between border-t border-stone-800 py-1">
              <span>{m.id}</span>
              <span className="text-stone-400">{mib(m.memory_bytes)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Daemon({ daemon }) {
  const alive = daemon?.alive;
  const label = alive === true ? "Alive" : alive === false ? "Down" : "Unknown";
  return (
    <div className="text-sm">
      <p>
        Status <span className="text-stone-200">{daemon?.status || "unknown"}</span>
      </p>
      <p className="mt-1 text-stone-400">Alive: {label}</p>
      {daemon?.detail ? <p className="mt-2 text-stone-500">{daemon.detail}</p> : null}
    </div>
  );
}

function Mcp({ tools }) {
  if (!tools || tools.length === 0) {
    return (
      <p className="text-sm text-stone-500">
        No MCP tools. <span className="text-stone-400">runa serve does not host MCP servers.</span>
      </p>
    );
  }
  return (
    <ul className="space-y-1 text-sm">
      {tools.map((name) => (
        <li key={name} className="border-t border-stone-800 py-1">
          {name}
        </li>
      ))}
    </ul>
  );
}

function Reasoning({ reasoning }) {
  const spent = Number(reasoning?.tokens_spent) || 0;
  const budget = reasoning?.active_budget_tokens;
  if (budget == null) {
    return (
      <div className="text-sm">
        <p>{int(spent)} reasoning tokens spent.</p>
        <p className="mt-1 text-stone-500">No active reasoning budget.</p>
      </div>
    );
  }
  const cap = Math.max(1, Number(budget));
  const pct = Math.min(100, (spent / cap) * 100);
  return (
    <div className="text-sm">
      <p>
        {int(spent)} spent, active budget {int(budget)} tokens.
      </p>
      <div className="mt-2 h-2 rounded bg-stone-800">
        <div className="h-2 rounded bg-amber-400" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
