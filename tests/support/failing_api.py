"""For local drills and tests only; never imported by the serving image."""

import services.api.app as app_module


def _fail(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("drill-injected-failure-7f3a")


app_module.run_predict = _fail
app = app_module.app
