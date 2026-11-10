"""Keep generated-file timestamps stable when their content is unchanged."""
def write_if_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
