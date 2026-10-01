import os
import pickle
import stat
import zipfile

import pytest

from diff_lab.adapters.jit_defects4j import _safe_member, extract_allowlisted
from diff_lab.adapters.pickle_worker import safe_load
from diff_lab.util import PolicyError


class Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_at22_malicious_pickle_blocked():
    with pytest.raises(pickle.UnpicklingError, match="blocked global"):
        safe_load(pickle.dumps(Evil()))


def test_builtin_only_pickle_loads():
    obj, resolved = safe_load(pickle.dumps([["a"], [1.0], ["m"], [{"added_code": {"x"}, "removed_code": set()}]]))
    assert obj[3][0]["added_code"] == {"x"} and resolved == []


@pytest.mark.parametrize("name", ["../evil.pkl", "/abs/evil.pkl", "data/../../x", "a\\b.pkl"])
def test_at22_path_traversal_rejected(name):
    with pytest.raises(PolicyError):
        _safe_member(zipfile.ZipInfo(name))


def test_at22_symlink_rejected():
    info = zipfile.ZipInfo("data/jitfine/link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(PolicyError, match="symlink"):
        _safe_member(info)


def test_at22_archive_with_traversal_member_rejected(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../escape.txt", "x")
    with pytest.raises(PolicyError):
        extract_allowlisted(z, tmp_path)
