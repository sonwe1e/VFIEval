from __future__ import annotations

import errno
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.server import _is_client_disconnect, _make_handler

from v13_test_utils import make_workspace


class _DisconnectingWriter:
    def __init__(self, exc: OSError) -> None:
        self.exc = exc
        self.write_count = 0

    def write(self, _payload: bytes) -> None:
        self.write_count += 1
        raise self.exc


class ServerClientDisconnectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.workspace, cls.db = make_workspace(cls._temporary_directory.name)
        cls.handler_type = _make_handler(cls.db, cls.workspace)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def _handler(self):
        handler = object.__new__(self.handler_type)
        handler.close_connection = False
        handler.command = "GET"
        handler.path = "/api/test"
        return handler

    def _run_at_request_boundary(self, handler, operation) -> None:
        def run_operation(request_handler) -> None:
            operation(request_handler)

        with mock.patch.object(
            BaseHTTPRequestHandler,
            "handle_one_request",
            new=run_operation,
        ):
            handler.handle_one_request()

    def _prepare_response_handler(self, range_header: str | None = None):
        handler = self._handler()
        handler.headers = {} if range_header is None else {"Range": range_header}
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.wfile = _DisconnectingWriter(BrokenPipeError(errno.EPIPE, "broken pipe"))
        return handler

    def test_disconnect_classifier_accepts_only_expected_socket_errors(self) -> None:
        expected = [
            BrokenPipeError(errno.EPIPE, "broken pipe"),
            ConnectionResetError(errno.ECONNRESET, "connection reset"),
            ConnectionAbortedError(errno.ECONNABORTED, "connection aborted"),
            OSError(errno.EPIPE, "broken pipe"),
            OSError(errno.ENOTCONN, "socket is not connected"),
            OSError(errno.ESHUTDOWN, "socket is shut down"),
        ]
        windows_broken_pipe = OSError("broken pipe")
        windows_broken_pipe.winerror = 109
        windows_shutdown = OSError("socket shutdown")
        windows_shutdown.winerror = 10058
        expected.extend([windows_broken_pipe, windows_shutdown])

        for exc in expected:
            with self.subTest(exc=repr(exc)):
                self.assertTrue(_is_client_disconnect(exc))

        access_denied = OSError(errno.EACCES, "access denied")
        access_denied.winerror = 5
        self.assertFalse(_is_client_disconnect(access_denied))
        self.assertFalse(_is_client_disconnect(RuntimeError("not a socket error")))

    def test_broken_pipe_before_client_io_is_still_an_internal_error(self) -> None:
        handler = self._handler()
        handler._json = mock.Mock()

        output = io.StringIO()
        with redirect_stdout(output):
            handler._error_internal(BrokenPipeError(errno.EPIPE, "worker pipe failed"))

        self.assertIn("BrokenPipeError: [Errno 32] worker pipe failed", output.getvalue())
        handler._json.assert_called_once_with(
            {"error": {"type": "InternalServerError", "message": "internal server error"}},
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        self.assertFalse(handler.close_connection)

    def test_json_disconnect_is_suppressed_at_request_boundary(self) -> None:
        handler = self._prepare_response_handler()

        output = io.StringIO()
        with redirect_stdout(output):
            self._run_at_request_boundary(
                handler,
                lambda request_handler: request_handler._json({"ok": True}),
            )

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(handler.wfile.write_count, 1)
        self.assertTrue(handler.close_connection)

    def test_typed_error_response_disconnect_is_suppressed_at_request_boundary(self) -> None:
        handler = self._prepare_response_handler()

        output = io.StringIO()
        with redirect_stdout(output):
            self._run_at_request_boundary(
                handler,
                lambda request_handler: request_handler._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid campaign",
                    "ValueError",
                ),
            )

        self.assertEqual(output.getvalue(), "")
        handler.send_response.assert_called_once_with(HTTPStatus.BAD_REQUEST)
        self.assertEqual(handler.wfile.write_count, 1)
        self.assertTrue(handler.close_connection)

    def test_post_handler_flush_disconnect_is_suppressed_at_request_boundary(self) -> None:
        handler = self._handler()
        handler.wfile = mock.Mock()
        handler.wfile.flush.side_effect = ConnectionResetError(
            errno.ECONNRESET,
            "connection reset",
        )

        self._run_at_request_boundary(
            handler,
            lambda request_handler: request_handler.wfile.flush(),
        )

        self.assertTrue(handler.close_connection)

    def test_file_and_range_disconnects_are_suppressed_at_request_boundary(self) -> None:
        payload_path = Path(self._temporary_directory.name) / "response.bin"
        payload_path.write_bytes(b"0123456789")

        for range_header, expected_status in (
            (None, HTTPStatus.OK),
            ("bytes=2-5", HTTPStatus.PARTIAL_CONTENT),
        ):
            with self.subTest(range_header=range_header):
                handler = self._prepare_response_handler(range_header)
                self._run_at_request_boundary(
                    handler,
                    lambda request_handler: request_handler._send_file(payload_path),
                )
                handler.send_response.assert_called_once_with(expected_status)
                self.assertEqual(handler.wfile.write_count, 1)
                self.assertTrue(handler.close_connection)

    def test_real_internal_error_is_logged_and_returns_500(self) -> None:
        handler = self._handler()
        handler._json = mock.Mock()

        output = io.StringIO()
        with redirect_stdout(output):
            handler._error_internal(RuntimeError("database failed"))

        self.assertIn("RuntimeError: database failed", output.getvalue())
        handler._json.assert_called_once_with(
            {"error": {"type": "InternalServerError", "message": "internal server error"}},
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        self.assertFalse(handler.close_connection)

    def test_disconnect_while_writing_500_preserves_only_original_error_log(self) -> None:
        handler = self._prepare_response_handler()

        output = io.StringIO()
        with redirect_stdout(output):
            self._run_at_request_boundary(
                handler,
                lambda request_handler: request_handler._error_internal(
                    RuntimeError("database failed")
                ),
            )

        self.assertIn("RuntimeError: database failed", output.getvalue())
        self.assertNotIn("BrokenPipeError", output.getvalue())
        self.assertTrue(handler.close_connection)

    def test_unrelated_oserror_is_not_suppressed(self) -> None:
        handler = self._handler()

        with self.assertRaises(OSError) as raised:
            self._run_at_request_boundary(
                handler,
                lambda _request_handler: (_ for _ in ()).throw(
                    OSError(errno.ENOSPC, "disk full")
                ),
            )

        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertFalse(handler.close_connection)


if __name__ == "__main__":
    unittest.main()
