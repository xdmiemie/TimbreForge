import argparse
import asyncio
import webbrowser

import requests
import uvicorn

from .api import app


def timbreforge_running(url):
    try:
        with requests.Session() as localhost:
            localhost.trust_env = False
            response = localhost.get(url + "api/health", timeout=2, allow_redirects=False)
            if response.status_code != 200:
                return False
            health = response.json()
    except (requests.RequestException, ValueError):
        return False
    if not isinstance(health, dict) or health.get("service") != "ready":
        return False
    if "name" in health:
        return health["name"] == app.title
    # Older running instances identify themselves through the fixed health defaults.
    return (health.get("architecture"), health.get("train_batch_size"),
            health.get("inference_steps")) == ("dit-1024-16", 16, 64)


def open_browser(url):
    try:
        if webbrowser.open(url):
            return
    except (webbrowser.Error, OSError):
        pass
    print(f"请在浏览器中打开 {url}", flush=True)


class BrowserServer(uvicorn.Server):
    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if self.started:
            await asyncio.to_thread(open_browser, f"http://{self.config.host}:{self.config.port}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--open-browser", action="store_true")
    arguments = parser.parse_args()
    if arguments.open_browser:
        url = f"http://127.0.0.1:{arguments.port}/"
        if timbreforge_running(url):
            open_browser(url)
            return
    server_class = BrowserServer if arguments.open_browser else uvicorn.Server
    server = server_class(uvicorn.Config(app, host="127.0.0.1", port=arguments.port,
                                         access_log=False, timeout_graceful_shutdown=5))
    app.state.server = server
    server.run()


if __name__ == "__main__":
    main()
