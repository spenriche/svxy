from flask import Flask, jsonify

from svxy_scanner import ScanError, scan_svxy_deep_itm_calls


app = Flask(__name__)


@app.get("/")
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

