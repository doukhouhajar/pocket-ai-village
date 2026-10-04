import os
# the stub-server test talks to 127.0.0.1
os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"]))
os.environ["no_proxy"] = os.environ["NO_PROXY"]
