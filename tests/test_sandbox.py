import pytest

from sicas.sandbox import DockerSandbox, LocalSandbox


def test_local_python_runs_and_times_out(tmp_path):
    sb = LocalSandbox()
    (tmp_path / "ok.py").write_text("print('hi')")
    (tmp_path / "loop.py").write_text("while True: pass")
    r = sb.run_python(["ok.py"], tmp_path, 5)
    assert r.returncode == 0 and "hi" in r.stdout
    r = sb.run_python(["loop.py"], tmp_path, 1)
    assert r.timed_out and r.returncode == 124


def test_local_shell_disabled_by_default(tmp_path):
    assert LocalSandbox().run_shell("echo pwned", tmp_path).returncode == 126


def test_local_is_flagged_non_isolated():
    assert LocalSandbox().isolated is False


@pytest.mark.skipif(not DockerSandbox.available(), reason="docker not available")
def test_docker_has_no_network_and_readonly_mount(tmp_path):
    sb = DockerSandbox()
    (tmp_path / "net.py").write_text(
        "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), 2)\n    print('NET')\n"
        "except OSError:\n    print('NONET')\n")
    assert "NONET" in sb.run_python(["net.py"], tmp_path, 20, readonly=True).stdout
    (tmp_path / "w.py").write_text("open('x.txt','w').write('1')")
    assert sb.run_python(["w.py"], tmp_path, 20, readonly=True).returncode != 0
