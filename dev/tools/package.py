#!/usr/bin/env python3
"""Build the runtime-only macOS archive and its Homebrew formula."""

import argparse
import base64
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260825/"
    "cpython-3.13.15%2B20260825-aarch64-apple-darwin-install_only_stripped.tar.gz"
)
PYTHON_SHA256 = "149038dd0c194c25d4616d7e42a35f67f2edee96412788f74115819b6a4c8548"
INSTALL_FILES = (
    "__init__.py",
    "launcher.py",
    "clients.py",
    "paths.py",
    "models.py",
    "service.py",
    "prepare_q8c.py",
    "convert_qwen38_q8.py",
    "catalog.py",
    "requirements.txt",
)
COMPLETION_FILES = ("models", "_splash", "splash.bash", "official-models.txt")
SERVER_FILES = (
    "__init__.py",
    "server.py",
    "backend.py",
    "constraints.py",
    "output.py",
    "frontend.py",
    "judgments.py",
    "diagnostics.py",
    "api_shapes.py",
    "tool_schema.py",
    "tokenization.py",
    "json_codec.py",
    "latency.py",
    "metrics.py",
    "errors.py",
    "runtime.py",
    "protocol.py",
    "images.py",
    "documents.py",
    "document_worker.py",
    "http_security.py",
    "thinking.py",
    "schema_validation.py",
    "crash_trace.py",
    "chat.html",
    "update_check.py",
    "runtime_profile.py",
)
NOTICE_FILES = (
    "LICENSE",
    "UPSTREAM.md",
    "ACKNOWLEDGEMENTS.md",
    "MODEL_ASSETS.md",
    "DEPENDENCY_LICENSE_INVENTORY.json",
    "SECURITY.md",
    "RELEASE_PROFILE.json",
    "THIRD_PARTY_NOTICES.md",
)
THIRD_PARTY_LICENSE_FILES = ("tokenizers-0.22.2-LICENSE",)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_runtime(destination, version):
    for folder, names in (
        ("install", INSTALL_FILES),
        ("install/completions", COMPLETION_FILES),
        ("server", SERVER_FILES),
        ("engine", ("splash", "splash.metallib")),
    ):
        (destination / folder).mkdir()
        source = ROOT / ("build" if folder == "engine" else folder)
        for name in names:
            shutil.copy2(source / name, destination / folder / name)
    for name in NOTICE_FILES:
        shutil.copy2(ROOT / name, destination / name)
    (destination / "third_party_licenses").mkdir()
    for name in THIRD_PARTY_LICENSE_FILES:
        shutil.copy2(
            ROOT / "third_party_licenses" / name,
            destination / "third_party_licenses" / name,
        )
    launcher = destination / "mltf"
    launcher.write_text('#!/bin/sh\nexport PYTHONDONTWRITEBYTECODE=1\nexec "$(dirname "$0")/python/bin/python3" -u "$(dirname "$0")/install/launcher.py" "$@"\n')
    launcher.chmod(0o755)
    (destination / "release.json").write_text(
        json.dumps(
            {
                "version": version,
                "binary_sha256": digest(destination / "engine/splash"),
                "metallib_sha256": digest(destination / "engine/splash.metallib"),
            },
            indent=2,
        )
        + "\n"
    )


def relocate_console_scripts(stage):
    """Make wheel entry points use the bundled Python after archive relocation."""
    scripts = {}
    binary_dir = stage / "python/bin"
    if not binary_dir.exists():
        return
    header = b"#!/bin/sh\n'''exec' \"$(dirname \"$0\")/python3\" \"$0\" \"$@\"\n' '''\n"
    for path in binary_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        original = path.read_bytes()
        first, separator, body = original.partition(b"\n")
        if (
            not separator
            or not first.startswith(b"#!")
            or str(stage).encode() not in first
            or b"python" not in first
        ):
            continue
        portable = header + body
        path.write_bytes(portable)
        scripts[path.resolve()] = portable
    site = stage / "python/lib/python3.13/site-packages"
    updated = set()
    for record in site.glob("*.dist-info/RECORD"):
        with record.open(newline="") as source:
            rows = list(csv.reader(source))
        changed = False
        for row in rows:
            if len(row) < 3:
                continue
            path = (site / row[0]).resolve()
            if path not in scripts and (
                not path.is_file() or path.parent != binary_dir.resolve()
            ):
                continue
            data = scripts[path] if path in scripts else path.read_bytes()
            checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            actual_hash = "sha256=" + checksum.rstrip(b"=").decode()
            actual_size = str(len(data))
            if row[1] != actual_hash or row[2] != actual_size:
                row[1], row[2] = actual_hash, actual_size
                changed = True
            if path in scripts:
                updated.add(path)
        if changed:
            with record.open("w", newline="") as output:
                csv.writer(output).writerows(rows)
    if updated != set(scripts):
        raise ValueError("a relocated console script has no wheel RECORD entry")


def formula(version, url, checksum, macos_min):
    # Inputs are encoded as Ruby double-quoted literals with the backslash,
    # the quote and the interpolation opener escaped, so no path or URL can
    # become executable Ruby; double quotes are what `brew audit` expects.
    def quote(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("#{", "\\#{")
        return '"' + escaped + '"'

    return f"""class MltfMacOSRequirement < Requirement
  fatal true
  satisfy(build_env: false) {{ OS.mac? && MacOS.full_version >= {quote(macos_min)} }}

  def message
    {quote(f"MLTF requires macOS {macos_min} or newer.")}
  end
end

class Mltf < Formula
  desc "Local inference engine for Apple silicon, built around the model"
  homepage "https://github.com/hojin12312/my-little-trie-forge"
  url {quote(url)}
  sha256 {quote(checksum)}
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on macos: :tahoe
  depends_on MltfMacOSRequirement

  def install
    libexec.install Dir["*"]
    (bin/"mltf").write <<~SH
      #!/bin/sh
      export PYTHONDONTWRITEBYTECODE=1
      exec "#{{opt_libexec}}/python/bin/python3" -u "#{{opt_libexec}}/install/launcher.py" "$@"
    SH
    chmod 0755, bin/"mltf"
    zsh_completion.install_symlink libexec/"install/completions/_splash" => "_mltf"
    bash_completion.install_symlink libexec/"install/completions/splash.bash" => "mltf"
  end

  def caveats
    <<~CAVEAT
      Serve a model:
        mltf serve --model local/Qwen3.8-27B-q8c --model-path /path/to/q8c
    CAVEAT
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/mltf --version")
    assert_match "serve", shell_output("#{{bin}}/mltf --help")
  end
end
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--macos-min", required=True, help="minimum macOS, e.g. 26.4")
    parser.add_argument(
        "--url", help="published tarball URL; defaults to GitHub Releases"
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error("invalid release version")
    dist = args.dist.resolve()
    dist.mkdir(parents=True, exist_ok=True)
    name = f"mltf-{args.version}-macos-arm64"
    archive = dist / f"{name}.tar.gz"
    if archive.exists():
        parser.error(f"release already exists: {archive}")
    cached = ROOT / "build/release/python-runtime.tar.gz"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists() or digest(cached) != PYTHON_SHA256:
        with tempfile.NamedTemporaryFile(dir=cached.parent) as download:
            with urllib.request.urlopen(PYTHON_URL, timeout=60) as response:
                shutil.copyfileobj(response, download)
            download.flush()
            if digest(Path(download.name)) != PYTHON_SHA256:
                raise ValueError("Python distribution checksum mismatch")
            shutil.copyfile(download.name, cached)
    with tempfile.TemporaryDirectory(prefix=".package-", dir=dist) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        stage_runtime(stage, args.version)
        with tarfile.open(cached) as python:
            python.extractall(stage, filter="data")
        python = stage / "python/bin/python3"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--no-compile",
                "-r",
                str(stage / "install/requirements.txt"),
            ],
            check=True,
        )
        relocate_console_scripts(stage)
        # Exercise imports and the public entry point from the actual staged
        # runtime, not the developer's virtualenv or source PYTHONPATH.
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import sys; sys.path.insert(0, '.'); "
                "import tokenizers, llguidance, numpy, PIL, transformers, huggingface_hub, pypdfium2; "
                "import server.server, install.launcher",
            ],
            cwd=stage,
            check=True,
        )
        subprocess.run(
            [str(python), "-B", str(stage / "install/launcher.py"), "--help"],
            cwd=stage,
            check=True,
        )
        packed = Path(temporary) / archive.name
        with tarfile.open(packed, "w:gz") as release:
            release.add(
                stage,
                arcname=name,
                filter=lambda item: (
                    None if "__pycache__" in Path(item.name).parts else item
                ),
            )
        packed.replace(archive)
    checksum = digest(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(checksum + "\n")
    url = (
        args.url
        or f"https://github.com/hojin12312/my-little-trie-forge/releases/download/v{args.version}/{archive.name}"
    )
    (dist / "mltf.rb").write_text(formula(args.version, url, checksum, args.macos_min))
    print(
        f"Built {archive}; run python3 dev/tools/package_brew.py build --version {args.version} "
        "before publishing the Homebrew formula."
    )


if __name__ == "__main__":
    main()
