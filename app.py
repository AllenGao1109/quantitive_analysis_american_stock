from flask import Flask, jsonify, render_template
import subprocess, json, os

app = Flask(__name__)

BASE_DIR = os.getcwd()
SCRIPT   = os.path.join(BASE_DIR, "spx_signal_v91.py")
BT_JSON  = os.path.join(BASE_DIR, "output", "backtest_result.json")

def run_script(args):
    cmd = ["conda", "run", "-n", "spx_strategy", "python", SCRIPT] + args
    try:
        out = subprocess.check_output(cmd, cwd=BASE_DIR, timeout=120).decode("utf-8").strip()
        return json.loads(out)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8") if e.stderr else "No stderr"
        return {"error": f"策略执行失败 (exit {e.returncode}):\n{stderr[:1000]}"}
    except json.JSONDecodeError:
        return {"error": "输出不是 JSON"}
    except Exception as e:
        return {"error": f"API错误: {str(e)}"}


@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/live")
def api_live():
    data = run_script(["--json"])
    return jsonify(data)

@app.get("/api/backtest")
def api_backtest():
    data = run_script(["--backtest", "--json"])
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
