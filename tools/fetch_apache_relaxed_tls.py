"""One-off runner: relax X509_STRICT (rejects the corporate TLS-inspection
proxy's non-RFC5280-compliant zenodo.org leaf cert, missing AKI) for this
process only, then call riskbench's normal fetch-apache. Not part of the
repo; riskbench's own pinned-MD5 check still guards integrity."""
import sys
import ssl

_orig = ssl.create_default_context


def _relaxed(*a, **kw):
    ctx = _orig(*a, **kw)
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


ssl.create_default_context = _relaxed

from riskbench.cli import main  # noqa: E402

sys.argv = ["riskbench", "fetch-apache", "--out", "data/raw/apachejit-v2"]
main()
