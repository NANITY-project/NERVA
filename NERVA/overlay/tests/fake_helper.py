#!/usr/bin/env python3
# a stand-in for a11y_helper.py used by perception_test.cpp: echoes {"id":N,"ok":true,"echo":<op>}; "slow" sleeps, "die" exits, "junk" prints garbage first
import json, sys, time
for line in sys.stdin:
    r = json.loads(line); op = r.get("op")
    if op == "slow": time.sleep(30)
    if op == "die": sys.exit(1)
    if op == "junk": print("this is not json at all", flush=True)
    print(json.dumps({"id": r["id"], "ok": True, "echo": op}), flush=True)
