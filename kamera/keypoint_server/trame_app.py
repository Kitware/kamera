import threading
from trame.app import get_server
from trame.ui.vuetify3 import VAppLayout
import trame.widgets.vuetify3 as v3
from trame.widgets import iframe

from kamera.keypoint_server.app import *  # noqa


server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller


with VAppLayout(server):
    with v3.VContainer(
        classes="pa-0 fill-height",
        fluid=True,
    ):
        with v3.VContainer(classes="fill-height", fluid=True):
            iframe.IFrame(
                classes="fill-height",
                style="width: 100%",
                fluid=True,
                src="http://localhost:8090/",
            )

if __name__ == "__main__":
    host = "localhost"
    port = 8090
    t = threading.Thread(target=run_keypoint_server, args=(host, port, "", "", ""))
    t.start()
    server.start()
