// Checks PyPI for the latest vLLM stable + nightly. Lazily refreshed.

interface ReleaseInfo {
  latestStable: string | null;
  latestNightly: string | null;
  fetchedAt: number;
  error: string | null;
}

let cache: ReleaseInfo | null = null;
const TTL_MS = 30 * 60 * 1000; // 30 minutes

async function fetchStable(): Promise<string | null> {
  try {
    const res = await fetch('https://pypi.org/pypi/vllm/json', {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const data = await res.json() as { info?: { version?: string } };
    return data.info?.version ?? null;
  } catch {
    return null;
  }
}

async function fetchNightly(): Promise<string | null> {
  try {
    const res = await fetch('https://wheels.vllm.ai/nightly/vllm/', {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const html = await res.text();
    // Parse simple `pip` index HTML for the most recent versioned wheel.
    const matches = html.match(/vllm-(\d[\w.+]*?)-cp\d/g);
    if (!matches || matches.length === 0) return null;
    const versions = matches
      .map(m => m.replace(/^vllm-/, '').replace(/-cp\d.*/, ''))
      .filter((v, i, a) => a.indexOf(v) === i)
      .sort();
    return versions[versions.length - 1] ?? null;
  } catch {
    return null;
  }
}

export async function getLatestRelease(force = false): Promise<ReleaseInfo> {
  const now = Date.now();
  if (!force && cache && now - cache.fetchedAt < TTL_MS) return cache;
  const [latestStable, latestNightly] = await Promise.all([fetchStable(), fetchNightly()]);
  cache = {
    latestStable,
    latestNightly,
    fetchedAt: now,
    error: latestStable || latestNightly ? null : 'Could not reach PyPI / wheels.vllm.ai',
  };
  return cache;
}

// Compare semver-ish version strings. Returns -1 if a < b, 0 if equal, 1 if a > b.
// Handles things like "0.20.2rc1.dev70+ge43a79128" vs "0.20.1".
export function compareVersions(a: string, b: string): number {
  const parse = (v: string): number[] => {
    const cleaned = v.replace(/^v/, '').split(/[+_]/)[0]; // strip build metadata
    const parts: number[] = [];
    const re = /(\d+)/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(cleaned)) !== null) parts.push(parseInt(m[1], 10));
    return parts;
  };
  const pa = parse(a);
  const pb = parse(b);
  const len = Math.max(pa.length, pb.length);
  for (let i = 0; i < len; i++) {
    const av = pa[i] ?? 0;
    const bv = pb[i] ?? 0;
    if (av !== bv) return av < bv ? -1 : 1;
  }
  return 0;
}
