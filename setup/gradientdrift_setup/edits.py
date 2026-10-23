"""Edits to the server's text files that change only the lines they mean to: comments, order, spacing
and everything else stay as they were. Each works on a TextFile held in memory; nothing is written
here (apply.py does that, after a backup)."""
import re


class TextFile:
    """A text file's lines, with its line ending and byte-order mark kept for writing back."""

    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            raw = f.read()
        self.bom = raw.startswith(b'\xef\xbb\xbf')
        text = raw[3:].decode('utf-8', errors='replace') if self.bom else raw.decode('utf-8', errors='replace')
        self.newline = '\r\n' if '\r\n' in text else '\n'
        self.final_newline = text.endswith(('\n', '\r'))
        self.lines = text.splitlines()

    def text(self):
        body = self.newline.join(self.lines) + (self.newline if self.final_newline or self.lines else '')
        return body

    def data(self):
        return (b'\xef\xbb\xbf' if self.bom else b'') + self.text().encode('utf-8')


# ---- Assetto Corsa .ini (server_cfg.ini, entry_list.ini) -----------------------------------------

def ini_set(tf, section, key, value):
    """Sets KEY=value in [SECTION]: the first such line is changed (its key's spelling kept), or a new
    line goes after the section's last key. The section must exist."""
    start = None
    for i, raw in enumerate(tf.lines):
        line = raw.strip()
        if line.startswith('[') and ']' in line and line[1:line.index(']')].strip().upper() == section.upper():
            start = i
            break
    if start is None:
        raise ValueError(f'{tf.path}: no [{section}] section')
    last_key = start
    for i in range(start + 1, len(tf.lines)):
        line = tf.lines[i].strip()
        if line.startswith('[') and ']' in line:
            break
        if line.startswith(';') or '=' not in line:
            continue
        name = line.split('=', 1)[0].strip()
        if name.upper() == key.upper():
            indent = tf.lines[i][:len(tf.lines[i]) - len(tf.lines[i].lstrip())]
            tf.lines[i] = f'{indent}{name}={value}'
            return
        last_key = i
    tf.lines.insert(last_key + 1, f'{key}={value}')


def append_block(tf, block):
    """Adds lines at the end, after one blank line."""
    while tf.lines and not tf.lines[-1].strip():
        tf.lines.pop()
    if tf.lines:
        tf.lines.append('')
    tf.lines += block.splitlines()
    tf.final_newline = True


# ---- extra_cfg.yml --------------------------------------------------------------------------------
# AssettoServer writes it as one main document (top-level keys at column 0, AiParams' keys indented
# under it) followed by one document per plugin ("--- !PluginConfiguration").

def _main_end(tf):
    """Index of the first line after the main document."""
    for i, line in enumerate(tf.lines):
        if i > 0 and line.startswith('---'):
            return i
    return len(tf.lines)


def _comment_split(value):
    """('value', '  # comment') of a YAML scalar line's value part."""
    m = re.match(r'^(.*?)(\s+#.*)?$', value)
    return m.group(1).strip(), m.group(2) or ''


def yaml_set_top(tf, key, value):
    """Sets a top-level key of the main document, or adds it at the main document's end."""
    end = _main_end(tf)
    pat = re.compile(rf'^{re.escape(key)}:(.*)$')
    for i in range(end):
        m = pat.match(tf.lines[i])
        if m:
            _, comment = _comment_split(m.group(1))
            tf.lines[i] = f'{key}: {value}{comment}'
            return
    tf.lines.insert(end, f'{key}: {value}')


def yaml_add_plugin(tf, name):
    """Adds a plugin to EnablePlugins (flow [A, B], empty, or a block list)."""
    end = _main_end(tf)
    for i in range(end):
        m = re.match(r'^EnablePlugins:(.*)$', tf.lines[i])
        if not m:
            continue
        value, comment = _comment_split(m.group(1))
        if value.startswith('['):
            items = [x.strip() for x in value.strip('[]').split(',') if x.strip()]
            if name not in items:
                items.append(name)
            tf.lines[i] = f'EnablePlugins: [{", ".join(items)}]{comment}'
            return
        if value in ('', '~', 'null'):
            j = i + 1
            items, indent = [], None
            while j < end and re.match(r'^\s*- ', tf.lines[j]):
                indent = tf.lines[j][:tf.lines[j].index('-')]
                items.append(tf.lines[j].split('-', 1)[1].strip())
                j += 1
            if indent is None:
                tf.lines[i] = f'EnablePlugins: [{name}]{comment}'
            elif name not in items:
                tf.lines.insert(j, f'{indent}- {name}')
            return
        raise ValueError(f'{tf.path}: EnablePlugins is "{value}", which setup does not know how to extend')
    tf.lines.insert(end, f'EnablePlugins: [{name}]')


def yaml_rename_plugin(tf, old, new):
    """Renames a plugin in EnablePlugins (flow or block list), keeping its place; dropped instead
    if the new name is already there."""
    end = _main_end(tf)
    for i in range(end):
        m = re.match(r'^EnablePlugins:(.*)$', tf.lines[i])
        if not m:
            continue
        value, comment = _comment_split(m.group(1))
        if value.startswith('['):
            items = [x.strip() for x in value.strip('[]').split(',') if x.strip()]
            items = [q for q in (new if q == old else q for q in items)]
            items = [q for k, q in enumerate(items) if q not in items[:k]]
            tf.lines[i] = f'EnablePlugins: [{", ".join(items)}]{comment}'
            return
        j, seen = i + 1, False
        while j < end and re.match(r'^\s*- ', tf.lines[j]):
            item = tf.lines[j].split('-', 1)[1].strip()
            if item == new:
                seen = True
            j += 1
        for k in range(i + 1, j):
            if tf.lines[k].split('-', 1)[1].strip() == old:
                if seen:
                    del tf.lines[k]
                else:
                    tf.lines[k] = tf.lines[k][:tf.lines[k].index('-')] + f'- {new}'
                return
        return


def yaml_rename_doc(tf, old_tag, new_tag):
    """Renames a document's tag (--- !Old -> --- !New), the document's contents as they are."""
    for i, line in enumerate(tf.lines):
        if line.startswith('---') and re.search(rf'{re.escape(old_tag)}(\s|$)', line):
            tf.lines[i] = line.replace(old_tag, new_tag, 1)
            return


def yaml_set_child(tf, parent, key, value):
    """Sets parent.key in the main document (parent a top-level mapping such as AiParams), at the
    parent's own indent; added as the parent's first key if missing; the parent added if missing."""
    end = _main_end(tf)
    start = next((i for i in range(end) if re.match(rf'^{re.escape(parent)}:\s*(#.*)?$', tf.lines[i])), None)
    if start is None:
        tf.lines[end:end] = [f'{parent}:', f'  {key}: {value}']
        return
    indent = None
    block_end = end
    for i in range(start + 1, end):
        line = tf.lines[i]
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if not line[0].isspace():
            block_end = i
            break
        own = line[:len(line) - len(line.lstrip())]
        if indent is None:
            indent = own
        if own == indent:
            m = re.match(rf'^{re.escape(indent)}{re.escape(key)}:(.*)$', line)
            if m:
                _, comment = _comment_split(m.group(1))
                tf.lines[i] = f'{indent}{key}: {value}{comment}'
                return
    tf.lines.insert(start + 1, f'{indent or "  "}{key}: {value}')


def yaml_plugin_doc(tf, tag, key, value):
    """Sets key in the plugin's own document (--- !Tag), adding the document at the end if missing."""
    start = next((i for i, l in enumerate(tf.lines) if l.startswith('---') and tag in l), None)
    if start is None:
        while tf.lines and not tf.lines[-1].strip():
            tf.lines.pop()
        tf.lines += [f'--- {tag}', f'{key}: {value}']
        tf.final_newline = True
        return
    end = next((i for i in range(start + 1, len(tf.lines)) if tf.lines[i].startswith('---')), len(tf.lines))
    for i in range(start + 1, end):
        m = re.match(rf'^{re.escape(key)}:(.*)$', tf.lines[i])
        if m:
            _, comment = _comment_split(m.group(1))
            tf.lines[i] = f'{key}: {value}{comment}'
            return
    tf.lines.insert(start + 1, f'{key}: {value}')


def _plugin_doc_key(tf, tag, key):
    """The line index of key in the plugin's own document (--- !Tag), or None."""
    start = next((i for i, l in enumerate(tf.lines) if l.startswith('---') and tag in l), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(tf.lines)) if tf.lines[i].startswith('---')), len(tf.lines))
    return next((i for i in range(start + 1, end) if re.match(rf'^{re.escape(key)}:', tf.lines[i])), None)


def yaml_plugin_doc_rename(tf, tag, old, new):
    """Renames key old to new in the plugin's own document, its value and comment as they are."""
    i = _plugin_doc_key(tf, tag, old)
    if i is not None:
        tf.lines[i] = new + tf.lines[i][len(old):]


def yaml_plugin_doc_remove(tf, tag, key):
    """Takes key out of the plugin's own document, with the lines indented under it (a list or a
    mapping as its value)."""
    i = _plugin_doc_key(tf, tag, key)
    if i is None:
        return
    j = i + 1
    while j < len(tf.lines) and tf.lines[j][:1].isspace() and tf.lines[j].strip():
        j += 1
    del tf.lines[i:j]
