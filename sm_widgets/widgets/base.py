import os
import time
import traceback
from typing import Optional
from uuid import uuid4

import orjson
from IPython.core.display import Javascript, display
from ipycallback import SlowTunnelWidget
from ipywidgets import HTML, Output
from loguru import logger
from sm.misc import deserialize_text


class BadRequest(Exception):
    pass


class BaseWidget:
    URL2HANDLER = {
        '/init/download_code': lambda self, params: self.jscode,
        '/init/done': lambda self, params: setattr(self, 'app_is_loaded', True)
    }

    RepeatUntilSuccess = r"""
    function repeatUntilSuccess(fn, timeout, maxTry) {
        if (fn() === true) {
            return;
        }
        if (maxTry === undefined) {
            maxTry = 10;
        }
        if (maxTry === 0) {
            console.error("Max retries error");
            alert("max retries error");
            throw new Error("Max retries error");
        } else {
            setTimeout(function () {
                repeatUntilSuccess(fn, timeout, maxTry - 1);
            }, timeout);
        }
    }
    """

    def __init__(self, app_name: str, dev: bool = False, embedded_error_log: bool = False):
        self.dev = dev
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if dev:
            infile = os.path.join(current_dir, f"../../webapps/{app_name}/build/static/js/main.js")
            if not os.path.exists(infile):
                infile = os.path.join(current_dir, f"jscodes/{app_name}.js.gz")
        else:
            infile = os.path.join(current_dir, f"jscodes/{app_name}.js.gz")
        self.jscode = deserialize_text(infile)
        self.app_id = str(uuid4())
        self.app_js_render_fn = "".join([s.capitalize() for s in app_name.replace("-", "_").split("_")]) + "." + ('renderDevApp' if self.dev else 'renderApp')
        self.tunnel = SlowTunnelWidget()
        self.tunnel.on_receive(self.communicate)
        self.app_is_loaded = False
        self.error_container = []
        if embedded_error_log:
            self.error_container.append(Output())

    def render(self, same_tab: bool = True, new_window: bool = False, shadow_dom: Optional[bool] = None):
        if shadow_dom is None:
            shadow_dom = True if same_tab else False

        setup_fn = 'setupApp' + self.app_id.replace("-", "")
        if same_tab:
            widgets = [self.tunnel, HTML(f'<div id="{self.app_id}"></div>'), *self.error_container]
            setup_html_container = "let win = window;"
        else:
            widgets = [self.tunnel, *self.error_container]
            if new_window:
                # important to use "about:blank", otherwise, you can't zoom
                open_window_code = f'window.open("about:blank", "app-{self.app_id}", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=window.outerWidth, height=window.outerHeight, top="+(screen.height)+", left="+(screen.width)");'
            else:
                open_window_code = f'window.open("about:blank", "app-{self.app_id}");'
            setup_html_container = f"""
            let win = {open_window_code}
            let title = win.document.createElement("title");
            title.innerHTML = '{self.__class__.__name__}';
            win.document.head.appendChild(title);
            
            let div = win.document.createElement("div");
            div.id = '{self.app_id}';
            win.document.body.appendChild(div);
            win.document.body.style = "padding: 8px 8px 0 8px;";
            """

        widgets.append(Javascript(f"""
        {self.RepeatUntilSuccess}
        if (window.IPyApps === undefined) {{
            window.IPyApps = new Map();
        }}
        function {setup_fn}() {{
            // get tunnel
            if (window.IPyCallback === undefined) {{
                return false;
            }}
            let tunnel = window.IPyCallback.get('{self.tunnel.tunnel_id}');
            if (tunnel === undefined) {{
                return false;
            }}
            
            // setup html container
            {setup_html_container}
            
            // use the tunnel first to send out the code, after the application is rendered, the listening function 
            // is going to be replaced by the listener in the application, so we don't have to worry.
            tunnel.on_receive(function (version, msg) {{
                let payload = JSON.parse(msg);
                if (payload.id !== '/init/download_code') {{
                    alert('invalid calling order. you need to set the source code first');
                    console.error("invalid call order. waiting for source code but get:", payload);
                    return;
                }}
                win.eval(payload.response);
                let shadowDOM = {str(shadow_dom).lower()};
                win.{self.app_js_render_fn}('{self.app_id}', tunnel, undefined, shadowDOM);
                window.IPyApps.set('{self.app_id}', win);
                tunnel.send_msg(JSON.stringify({{ url: '/init/done', params: null, id: '/init/done' }}));
            }});
            tunnel.send_msg(JSON.stringify({{ url: '/init/download_code', params: null, id: '/init/download_code' }}));
            return true;
        }}
        repeatUntilSuccess({setup_fn}, 50, 10);
        """))
        display(*widgets)
        return self

    def communicate(self, version: int, msg: str):
        try:
            payload = orjson.loads(msg)
            request_id, url, params = payload['id'], payload['url'], payload['params']
            resp = self.URL2HANDLER[url](self, params)
            is_success = True
        except BadRequest as e:
            is_success = False
            resp = str(e)
        except Exception as e:
            # having it here since default Jupyter Notebook doesn't show the log
            if not hasattr(BaseWidget, '_has_inited_loguru_'):
                setattr(BaseWidget, '_has_inited_loguru_', True)
                logger.add("/tmp/notebook_app.log")
            logger.exception("Error while handling incoming messasges")

            if len(self.error_container) > 0:
                with self.error_container[0]:
                    traceback.print_exc()
            raise

        self.tunnel.send_msg(orjson.dumps({
            "type": "response",
            "id": request_id,
            "success": is_success,
            "response": resp
        }).decode())

    def wait_for_app_ready(self, timeout_seconds: int):
        # pulling every 100ms
        idle = 0.1
        n_pull = int(timeout_seconds / idle)
        for i in range(n_pull):
            if self.app_is_loaded:
                return True
            time.sleep(idle)
        raise Exception("Timeout")

    @classmethod
    def register_handler(cls, url: str):
        def wrapper_fn(func):
            cls.URL2HANDLER[url] = func
            return func
        return wrapper_fn
