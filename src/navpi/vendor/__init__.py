"""Package-local OpenPI modifications, no writes to transformers/site-packages."""
from importlib.metadata import version

if version("transformers") != "4.53.2":
    raise RuntimeError("NavPi's vendored model components require transformers==4.53.2")
