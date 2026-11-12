export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch('/api/v1'+path, body === undefined ? undefined : {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({message: response.statusText}));
    throw new Error(`${error.code ?? response.status}: ${error.message}`);
  }
  return response.json() as Promise<T>;
}
export async function download(path: string, filename: string, body?: unknown) {
  const response = await fetch('/api/v1'+path, body === undefined ? undefined : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (!response.ok) throw new Error(await response.text());
  const url = URL.createObjectURL(await response.blob());
  const a = document.createElement('a'); a.href=url; a.download=filename; a.click();
  setTimeout(()=>URL.revokeObjectURL(url), 2000);
}
