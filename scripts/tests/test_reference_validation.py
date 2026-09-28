import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_reference import ensure_asset, verify


class ReferenceAssets(unittest.TestCase):
    def test_missing_and_corrupt_assets_fail_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
            with self.assertRaises(ValueError):
                ensure_asset(root, 'asset', expected, 'invalid-url', False)
            (root / 'asset').write_bytes(b'xyz')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                verify(root / 'asset', expected)
            (root / 'asset').write_bytes(b'abc')
            self.assertEqual(ensure_asset(root, 'asset', expected, 'invalid-url', False), expected['sha256'])

    def test_bad_download_never_replaces_existing_cache(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'asset').write_bytes(b'old')
            expected = dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
            for payload in [b'xyz', b'oversized']:
                with patch('validate_reference.open_asset', return_value=io.BytesIO(payload)):
                    with self.assertRaises(ValueError):
                        ensure_asset(root, 'asset', expected, 'https://example.invalid/asset', True)
                self.assertEqual((root / 'asset').read_bytes(), b'old')
                self.assertFalse((root / 'asset.part').exists())

    def test_verified_download_replaces_cache(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
            with patch('validate_reference.open_asset', return_value=io.BytesIO(b'abc')):
                self.assertEqual(ensure_asset(root, 'asset', expected, 'https://example.invalid/asset', True), expected['sha256'])
            self.assertEqual((root / 'asset').read_bytes(), b'abc')

    def test_permanent_redirect_download_on_python_310(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/first':
                    self.send_response(308)
                    self.send_header('Location', '/final')
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'abc')
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                expected = dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
                url = f'http://127.0.0.1:{server.server_port}/first'
                self.assertEqual(ensure_asset(Path(directory), 'asset', expected, url, True), expected['sha256'])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_manifest_matches_committed_oracles(self):
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((root / 'scripts/reference-model.json').read_text())
        model = json.loads((root / 'crates/mini-vllm-model/tests/fixtures/qwen2.5-0.5b-reference.json').read_text())
        chat = json.loads((root / 'crates/mini-vllm-tokenizer/tests/fixtures/chat-reference.json').read_text())
        self.assertEqual(manifest['files']['tokenizer.json']['sha256'], chat['tokenizer_sha256'])
        self.assertEqual(manifest['files']['config.json']['sha256'], model['config_sha256'])
        for name, value in model['weights_sha256'].items():
            self.assertEqual(manifest['files'][name]['sha256'], value)
