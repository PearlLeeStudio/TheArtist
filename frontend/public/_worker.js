// Cloudflare Pages serves the editor; a separate Python host runs the models.
const unavailable = (detail, status = 503) => Response.json(
  { detail }, { status, headers: { 'Cache-Control': 'no-store' } },
);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== '/api' && !url.pathname.startsWith('/api/')) {
      return env.ASSETS.fetch(request);
    }

    let backend;
    try {
      backend = new URL(env.BACKEND_ORIGIN);
      if (backend.protocol !== 'https:' || backend.username || backend.password
          || backend.pathname !== '/' || backend.search || backend.hash
          || backend.origin === url.origin) throw new Error('Invalid origin');
    } catch {
      return unavailable('The music server is not connected yet. Please try again later.');
    }
    backend.pathname = url.pathname;
    backend.search = url.search;
    const headers = new Headers();
    for (const name of ['Accept', 'Content-Type']) {
      const value = request.headers.get(name);
      if (value !== null) headers.set(name, value);
    }
    try {
      const response = await fetch(backend, {
        method: request.method,
        headers,
        body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
        redirect: 'manual',
        signal: request.signal,
      });
      if (response.status >= 300 && response.status < 400) {
        return unavailable('The music server returned an unexpected redirect.', 502);
      }
      const outgoing = new Headers(response.headers);
      outgoing.set('Cache-Control', 'no-store');
      outgoing.delete('Set-Cookie');
      return new Response(response.body, {
        status: response.status, statusText: response.statusText, headers: outgoing,
      });
    } catch {
      return unavailable('The music server is unavailable. Please try again later.', 502);
    }
  },
};
