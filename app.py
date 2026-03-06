from pathlib import Path

from flask import Flask, Response, jsonify

from svxy_scanner import ScanError, scan_svxy_deep_itm_calls


app = Flask(__name__)
INDEX_FILE = Path(__file__).with_name("index.html")


@app.get("/")
def home() -> Response:
    return Response(INDEX_FILE.read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/api/svxy")
def get_svxy_rankings():
    try:
        response = jsonify(scan_svxy_deep_itm_calls())
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response
    except ScanError as exc:
        response = jsonify({"error": str(exc)})
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response, 502
    except Exception as exc:  # pragma: no cover
        response = jsonify({"error": f"Unexpected failure: {exc}"})
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response, 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
