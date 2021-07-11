from typing import Callable, Any, Optional, Union
from typing import List, TypedDict
from uuid import uuid4

import ipywidgets
import orjson
from IPython.core.display import Javascript
from IPython.core.display import display
from ipywidgets import HTML

from sm_widgets.widgets.base import BaseWidget


class _Slider(BaseWidget):
    pass


class Slider(_Slider):
    def __init__(self, app: BaseWidget, app_render_fn: Union[Callable, str], dev: bool = False, embedded_error_log: bool = False):
        super().__init__("slider", dev=dev, embedded_error_log=embedded_error_log)
        self.app = app
        self.app_render_fn = getattr(app, app_render_fn) if isinstance(app_render_fn, str) else app_render_fn
        self.index = 0
    
    def render(self, same_tab: bool = True, new_window: bool = False, shadow_dom: Optional[bool] = None):
        if shadow_dom is None:
            shadow_dom = True if same_tab else False

        setup_fn = 'setupApp' + self.app_id.replace("-", "")
        self.app.render(same_tab, new_window, shadow_dom)
        widgets = [self.tunnel, *self.error_container]
        widgets.append(Javascript(f"""
            {self.RepeatUntilSuccess}
            function {setup_fn}() {{
                if (window.IPyCallback === undefined || window.IPyApps === undefined) {{
                    return false;
                }}
                let tunnel = window.IPyCallback.get('{self.tunnel.tunnel_id}');
                if (tunnel === undefined) {{
                    return false;
                }}
                
                let appwin = window.IPyApps.get('{self.app.app_id}');
                if (appwin === undefined) {{
                    return false;
                }}
                
                let container = appwin.document.getElementById('{self.app.app_id}');
                if (container === null) {{
                    return false;
                }}
                
                let div = appwin.document.createElement("div");
                div.id = '{self.app_id}';
                div.style = "margin-bottom: 8px";
                container.parentElement.prepend(div);
                
                // use the tunnel first to send out the code, after the application is rendered, the listening function 
                // is going to be replaced by the listener in the application, so we don't have to worry.
                tunnel.on_receive(function (version, msg) {{
                    let payload = JSON.parse(msg);
                    if (payload.id !== '/init/download_code') {{
                        alert('invalid calling order. you need to set the source code first');
                        console.error("invalid call order", payload);
                        return;
                    }}
                    appwin.eval(payload.response);
                    appwin.{self.app_js_render_fn}('{self.app_id}', tunnel);
                    window.IPyApps.set('{self.app_id}', appwin);
                    tunnel.send_msg(JSON.stringify({{ url: '/init/done', params: null, id: '/init/done' }}));
                }});
                tunnel.send_msg(JSON.stringify({{ url: '/init/download_code', params: null, id: '/init/download_code' }}));
                return true;
            }}
            repeatUntilSuccess({setup_fn}, 50, 10);
        """))
        display(*widgets)
        return self
    
    def set_data(self, lst: List[TypedDict('SliderAppLst', description=str, args=Any)], start_index: int=0):
        self.wait_for_app_ready(5)
        self.lst = lst
        self.index = start_index
        self.tunnel.send_msg(orjson.dumps([
            {
                "type": "wait_for_client_ready"
            },
            {
                "type": "set_props",
                "props": {
                    "min": 0,
                    "max": len(lst) - 1,
                    "index": self.index,
                    "description": lst[self.index]['description']
                }
            }
        ]).decode())
        self.app_render_fn(*lst[self.index]['args'])

    @_Slider.register_handler("/view")
    def view(self, params: dict):
        self.index = params['index']
        self.tunnel.send_msg(orjson.dumps([
            {
                "type": "set_props",
                "props": {
                    "description": self.lst[self.index]['description']
                }
            }
        ]).decode())
        self.app_render_fn(*self.lst[self.index]['args'])


class DisplayShell:
    """A basic application to make default jupyter application work with slider"""
    def __init__(self, render_fn: Callable[[Any], None]):
        self.output = ipywidgets.Output()
        self.render_fn = render_fn
        self.app_id = str(uuid4())

    def render(self, same_tab: bool = True, new_window: bool = False, shadow_dom: Optional[bool] = None):
        assert same_tab is True and new_window is False
        display(HTML(f'<div id="{self.app_id}"></div>'), self.output)
        display(Javascript(f"""
        if (window.IPyApps === undefined) {{
            window.IPyApps = new Map();
        }}
        window.IPyApps.set('{self.app_id}', window);
        """))

    def set_data(self, item):
        with self.output:
            self.output.clear_output()
            self.render_fn(item)
