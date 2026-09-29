"""Run the local lab in installed Chromium via CDP; close server/browser on exit.

Requires websocket-client. Uses a headless CPU browser on the current host, not
a measurement on a consumer laptop or of a WebGPU adapter.
"""
import argparse
import functools
import http.server
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import urllib.request
import websocket


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--chromium",required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):
            pass
    server=http.server.ThreadingHTTPServer(("127.0.0.1",0),functools.partial(Quiet,directory=str(root)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    process=None
    try:
        with tempfile.TemporaryDirectory(prefix="dex-chromium-") as profile:
            process=subprocess.Popen([args.chromium,"--headless=new","--no-sandbox","--disable-gpu",
                "--disable-dev-shm-usage","--remote-debugging-port=0",f"--user-data-dir={profile}","about:blank"],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            portfile=Path(profile)/"DevToolsActivePort"
            for _ in range(100):
                if portfile.exists():break
                time.sleep(.1)
            port=int(portfile.read_text().splitlines()[0])
            pages=[]
            for _ in range(100):
                pages=json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                if pages:break
                time.sleep(.1)
            if not pages:raise RuntimeError("Chromium did not create a page")
            ws=websocket.create_connection(pages[0]["webSocketDebuggerUrl"],timeout=180,suppress_origin=True)
            serial=0
            def call(method,params):
                nonlocal serial
                serial+=1
                ws.send(json.dumps({"id":serial,"method":method,"params":params}))
                while True:
                    result=json.loads(ws.recv())
                    if result.get("id")==serial:
                        if "error" in result:raise RuntimeError(result["error"])
                        return result["result"]
            call("Page.navigate",{"url":f"http://127.0.0.1:{server.server_port}/web/"})
            for _ in range(100):
                result=call("Runtime.evaluate",{"expression":"window.dexReady === true","returnByValue":True})
                if result.get("result",{}).get("value"):break
                time.sleep(.1)
            def evaluate(expr):
                result=call("Runtime.evaluate",{"expression":expr,"returnByValue":True,"awaitPromise":True,"timeout":170000})
                if "exceptionDetails" in result:raise RuntimeError(result["exceptionDetails"])
                return result["result"]["value"]
            numeric=evaluate("window.runNumericBenchmark()")
            neural=evaluate("window.runNeuralBenchmark()")
            report={"host":"Xeon D-1521, headless Chromium, GPU disabled", "numeric":numeric,"neural":neural}
            (root/"artifacts"/"browser-measurements.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report,indent=2))
            assert numeric["contains_reference"] and numeric["argmax_certified"]
            assert neural["parity_passed"]
            ws.close()
            process.terminate()
            process.wait(timeout=10)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        server.shutdown()
        server.server_close()


if __name__=="__main__":
    main()
