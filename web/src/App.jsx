import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

const empty = {
  journal: "",
  now: 0,
  summary: {
    requests: 0,
    prompt_tokens: 0,
    generated_tokens: 0,
    reasoning_tokens: 0,
    avg_tg_toks_per_s: 0,
  },
  by_model: [],
  by_source: [],
  by_day: [],
  by_hour: [],
  recent: [],
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

function clock(ts) {
  const n = Number(ts);
  if (!Number.isFinite(n) || n <= 0) return "—";
  return new Date(n * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function hourLabel(unix) {
  const d = new Date(unix * 1000);
  return d.toLocaleTimeString(undefined, { hour: "2-digit" });
}

function hours24(rows, nowSec) {
  const end = Math.floor((nowSec || Date.now() / 1000) / 3600) * 3600;
  const map = new Map((rows || []).map((row) => [Number(row.hour), Number(row.generated_tokens) || 0]));
  const out = [];
  for (let i = 23; i >= 0; i -= 1) {
    const hour = end - i * 3600;
    out.push({ hour, generated: map.get(hour) || 0 });
  }
  return out;
}

async function loadSnapshot() {
  const res = await fetch("/api/snapshot");
  if (!res.ok) throw new Error(`snapshot ${res.status}`);
  return res.json();
}

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="12" fill="#1C1412" />
      <path d="M32 12 L32 48" stroke="#F07050" strokeWidth="3" strokeLinecap="round" fill="none" />
      <path d="M32 20 L18 32" stroke="#E85A3C" strokeWidth="2.8" strokeLinecap="round" fill="none" />
      <path d="M32 20 L46 32" stroke="#E85A3C" strokeWidth="2.8" strokeLinecap="round" fill="none" />
      <path d="M32 48 L26 54 L32 50 L38 54 Z" fill="#FF9A6A" />
      <circle cx="32" cy="48" r="3.5" fill="#FFB088" />
      <circle cx="32" cy="47" r="1.6" fill="#FFF0E8" />
    </svg>
  );
}

export default function App() {
  const queryClient = useQueryClient();
  const [link, setLink] = useState("connecting");
  const [theme, setTheme] = useState(() => localStorage.getItem("runa-theme") || "");
  const query = useQuery({ queryKey: ["snapshot"], queryFn: loadSnapshot });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "light" || theme === "dark") {
      root.dataset.theme = theme;
      localStorage.setItem("runa-theme", theme);
    } else {
      delete root.dataset.theme;
      localStorage.removeItem("runa-theme");
    }
  }, [theme]);

  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
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
  const summary = snap.summary ?? empty.summary;
  const bars = hours24(snap.by_hour, snap.now);
  const peak = Math.max(1, ...bars.map((b) => b.generated));
  const nextTheme = theme === "dark" ? "light" : theme === "light" ? "" : "dark";
  const themeLabel = theme === "dark" ? "dark" : theme === "light" ? "light" : "auto";

  return (
    <main className="shell">
      <header className="top">
        <div className="brand">
          <Mark />
          <div>
            <p className="eyebrow">runa</p>
            <h1>Usage</h1>
          </div>
        </div>
        <div className="tools">
          <button type="button" className="ghost" onClick={() => setTheme(nextTheme)}>
            {themeLabel}
          </button>
          <p className="live">
            <span className={link === "live" ? "dot on" : "dot"} />
            {link}
            {query.isError ? ` · ${query.error.message}` : ""}
          </p>
        </div>
      </header>

      <section className="tiles">
        <article className="tile">
          <span>Requests</span>
          <strong>{int(summary.requests)}</strong>
          <em>finished generations</em>
        </article>
        <article className="tile">
          <span>Prompt</span>
          <strong>{int(summary.prompt_tokens)}</strong>
          <em>tokens in</em>
        </article>
        <article className="tile">
          <span>Generated</span>
          <strong>{int(summary.generated_tokens)}</strong>
          <em>tokens out</em>
        </article>
        <article className="tile">
          <span>Reasoning</span>
          <strong>{int(summary.reasoning_tokens)}</strong>
          <em>inside the answer</em>
        </article>
        <article className="tile">
          <span>Decode</span>
          <strong>{num(summary.avg_tg_toks_per_s)}</strong>
          <em>tok/s, weighted</em>
        </article>
      </section>

      <section className="grid">
        <article className="card">
          <h2>Last 24 hours</h2>
          <div className="bars" aria-hidden="true">
            {bars.map((bar) => (
              <div className="bar" key={bar.hour} title={`${hourLabel(bar.hour)} · ${int(bar.generated)}`}>
                <i
                  className={bar.generated ? "" : "empty"}
                  style={bar.generated ? { height: `${Math.max(6, (bar.generated / peak) * 100)}%` } : undefined}
                />
              </div>
            ))}
          </div>
          <div className="axis">
            <span>{hourLabel(bars[0]?.hour)}</span>
            <span>now</span>
          </div>
        </article>
        <article className="card">
          <h2>Sources</h2>
          {(snap.by_source || []).length === 0 ? (
            <p className="empty">Nothing recorded yet. `runa run`, `chat`, and `serve` append here.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th className="num">Requests</th>
                  <th className="num">Generated</th>
                </tr>
              </thead>
              <tbody>
                {snap.by_source.map((row) => (
                  <tr key={row.source}>
                    <td>
                      <span className="chip">{row.source}</span>
                    </td>
                    <td className="num">{int(row.requests)}</td>
                    <td className="num">{int(row.generated_tokens)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
      </section>

      <section className="grid">
        <article className="card">
          <h2>Models</h2>
          {(snap.by_model || []).length === 0 ? (
            <p className="empty">Models show up after the first finished generation.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num">Requests</th>
                  <th className="num">Generated</th>
                  <th className="num">tok/s</th>
                </tr>
              </thead>
              <tbody>
                {snap.by_model.map((row) => (
                  <tr key={row.model}>
                    <td className="model" title={row.model}>
                      {row.model || "—"}
                    </td>
                    <td className="num">{int(row.requests)}</td>
                    <td className="num">{int(row.generated_tokens)}</td>
                    <td className="num">{num(row.avg_tg_toks_per_s)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
        <article className="card">
          <h2>Recent</h2>
          {(snap.recent || []).length === 0 ? (
            <p className="empty">The feed updates as soon as a request finishes.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Source</th>
                  <th className="num">Out</th>
                </tr>
              </thead>
              <tbody>
                {snap.recent.map((row) => (
                  <tr key={`${row.ts}-${row.model}-${row.source}`}>
                    <td>
                      {clock(row.ts)}
                      <div className="model" title={row.model}>
                        {row.model || "—"}
                      </div>
                    </td>
                    <td>
                      <span className="chip">{row.source}</span>
                    </td>
                    <td className="num">{int(row.generated_tokens)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
      </section>

      <p className="foot">{snap.journal ? `journal ${snap.journal}` : "journal"}</p>
    </main>
  );
}
