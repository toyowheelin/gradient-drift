"""Assetto Corsa .ini files (server_cfg.ini, entry_list.ini): read them the way the game does
(section and key names case-insensitive, lines starting with ';' are comments - not a ';' later
in the line, which server_cfg.ini's CARS= uses as its separator - and the first value wins) while keeping every
line, so a later edit can change only what it means to."""


class AcIni:
    def __init__(self, path):
        self.path = path
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            self.lines = f.read().splitlines()

    def sections(self):
        """[(NAME, {KEY: value})] in file order."""
        out, current = [], None
        for raw in self.lines:
            line = raw.strip()
            if line.startswith(';'):
                continue
            if line.startswith('[') and ']' in line:
                current = (line[1:line.index(']')].strip().upper(), {})
                out.append(current)
            elif '=' in line and current is not None:
                key, value = line.split('=', 1)
                current[1].setdefault(key.strip().upper(), value.strip())
        return out

    def section(self, name):
        for sec, values in self.sections():
            if sec == name.upper():
                return values
        return {}
