// AES-256-GCM encryption for user-supplied secrets stored in the DB.
// The key is derived from SUPABASE_SERVICE_ROLE_KEY (already a server secret),
// so we don't introduce a new env var. Ciphertext format:
//   v1.<base64url(iv)>.<base64url(ciphertext+tag)>
// Server-only. NEVER import from client code.
import { webcrypto } from "node:crypto";

const subtle = webcrypto.subtle;

let cachedKey: CryptoKey | null = null;

async function getKey(): Promise<CryptoKey> {
  if (cachedKey) return cachedKey;
  const material = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!material) throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY for secret encryption");
  const hash = await subtle.digest("SHA-256", new TextEncoder().encode("tirthoai:user-models:v1:" + material));
  cachedKey = await subtle.importKey("raw", hash, { name: "AES-GCM" }, false, ["encrypt", "decrypt"]);
  return cachedKey;
}

function b64u(buf: ArrayBuffer | Uint8Array): string {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromB64u(s: string): Uint8Array {
  const padded = s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4);
  const bin = atob(padded);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

export async function encryptSecret(plain: string): Promise<string> {
  if (!plain) return "";
  const key = await getKey();
  const iv = webcrypto.getRandomValues(new Uint8Array(12));
  const ct = await subtle.encrypt({ name: "AES-GCM", iv }, key, new TextEncoder().encode(plain));
  return `v1.${b64u(iv)}.${b64u(ct)}`;
}

export async function decryptSecret(blob: string): Promise<string> {
  if (!blob) return "";
  const parts = blob.split(".");
  if (parts.length !== 3 || parts[0] !== "v1") {
    throw new Error("Invalid ciphertext format");
  }
  const iv = fromB64u(parts[1]);
  const ct = fromB64u(parts[2]);
  const key = await getKey();
  const pt = await subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
  return new TextDecoder().decode(pt);
}

/** Return last 4 chars of the plaintext key for UI hinting, without exposing it. */
export function maskHint(plain: string): string {
  if (!plain || plain.length < 4) return "••••";
  return "••••" + plain.slice(-4);
}
