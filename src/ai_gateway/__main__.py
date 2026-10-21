import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(prog="ai-gateway")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default=None, help="path to gateway.toml")
    a = p.parse_args()

    from .app import create_app
    from .backends import load_registry

    uvicorn.run(create_app(load_registry(a.config)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
