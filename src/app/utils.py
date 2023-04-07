import logging
import sys
import time
from contextlib import contextmanager
from io import StringIO
from threading import current_thread

import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b + "\r\n")
                output_func(buffer.getvalue() + "")
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    "this will show the prints"
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    "This will show the logging"
    with st_redirect(sys.stderr, dst):
        yield
