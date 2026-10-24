"""Asking and telling in a terminal, the same on Windows and Linux. With prompt_toolkit (and a real
terminal) a choice is a list picked with the arrow keys and a folder is typed with Tab completing
it; without either it falls back to numbered lists and plain input(), so piped input still works.
GNU readline is never used (it is GPL, and the packaged program leaves it out)."""
import os
import shutil
import sys
import textwrap
import threading
import time

try:
    from prompt_toolkit import print_formatted_text, prompt
    from prompt_toolkit.completion import PathCompleter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.shortcuts import choice as _choice
    from prompt_toolkit.validation import ValidationError, Validator
    HAVE_TOOLKIT = True
except ImportError:
    HAVE_TOOLKIT = False


class Stop(Exception):
    """The operator quit (Ctrl+C, or the end of piped input)."""


def _marks():
    try:
        '✔⚠✖'.encode(sys.stdout.encoding or 'ascii')
        return '✔', '⚠', '✖'
    except (UnicodeEncodeError, LookupError):
        return 'ok', '!', 'X'


OK, WARN, BAD = _marks()
_STYLE = {OK: 'ansigreen', WARN: 'ansiyellow', BAD: 'ansired'}


def rich():
    """Arrow keys and Tab completion: prompt_toolkit, and a terminal at both ends."""
    return HAVE_TOOLKIT and sys.stdin.isatty() and sys.stdout.isatty()


def _wrap(text):
    """A line too wide for the terminal, folded under its own start (past its indent, mark or
    bullet), so a long change or problem reads as one item."""
    width = shutil.get_terminal_size((100, 24)).columns - 1
    if len(text) <= width or '\n' in text:
        return [text]
    body = text.lstrip()
    lead = len(text) - len(body)
    for start in (OK + ' ', WARN + ' ', BAD + ' ', '- ', '! '):
        if body.startswith(start):
            lead += len(start)
            break
    return textwrap.wrap(text, width, subsequent_indent=' ' * lead, break_on_hyphens=False) or [text]


def say(text='', style='', fold=True):
    """A line out, folded to the terminal's width (fold=False: a line to copy, a command, stays
    whole). A line starting with one of the marks gets its colour."""
    for line in (_wrap(text) if fold else [text]):
        _say(line, style)


def _say(text, style):
    if rich():
        for mark, colour in _STYLE.items():
            if text.lstrip().startswith(mark):
                pad = text[:len(text) - len(text.lstrip())]
                print_formatted_text(FormattedText([('', pad), (colour, mark), (style, text.lstrip()[len(mark):])]))
                return
        print_formatted_text(FormattedText([(style, text)]))
    else:
        print(text, flush=True)


def heading(step, total, text):
    say()
    say(f'{step}/{total}  {text}', 'bold')


def hint(text):
    """Help under a question, in the quieter colour (the window's grey hint lines), folded to the
    terminal's width (at most 100 columns) as one paragraph."""
    width = min(100, shutil.get_terminal_size((100, 24)).columns - 1)
    for line in textwrap.wrap(' '.join(text.split()), width, initial_indent=' ' * 5, subsequent_indent=' ' * 5,
                              break_on_hyphens=False, break_long_words=False):
        _say(line, 'ansibrightblack' if rich() else '')


def _input(message):
    try:
        return input(message)
    except (EOFError, KeyboardInterrupt):
        print()
        raise Stop() from None


def choose(question, options, default=None):
    """One of options [(value, label)]; Enter takes the default (else the first)."""
    if default is None or default not in [v for v, _ in options]:
        default = options[0][0]
    if rich():
        try:
            return _choice(message=FormattedText([('bold', f'     {question}')]), options=options, default=default,
                           bottom_toolbar=' Up/Down to move, Enter to pick, Ctrl+C to quit ')
        except (KeyboardInterrupt, EOFError):
            raise Stop() from None
    say(f'     {question}')
    for n, (value, label) in enumerate(options, 1):
        say(f'       {n}) {label}')
    first = [v for v, _ in options].index(default) + 1
    while True:
        answer = _input(f'     Pick [{first}]: ').strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        say(f'     Type a number from 1 to {len(options)}.')


def _clean_path(text):
    text = text.strip().strip('"').strip("'")
    return os.path.normpath(os.path.expanduser(text)) if text else ''


def ask_path(question, check, refusal, default='', allow_empty=False):
    """A folder, typed with Tab completing it. check(path) says whether it will do; refusal(path)
    is what to say when it will not. allow_empty: an empty answer is an answer ('')."""
    if rich():
        class Explain(Validator):        # the refusal names what is wrong with this path
            def validate(self, document):
                path = _clean_path(document.text)
                if not path and allow_empty:
                    return
                if not path or not check(path):
                    raise ValidationError(message=refusal(path) if path else 'Type a folder (Tab completes it).',
                                          cursor_position=len(document.text))
        try:
            text = prompt(FormattedText([('bold', f'     {question} ')]), default=default,
                          completer=PathCompleter(only_directories=True, expanduser=True),
                          complete_while_typing=False, validator=Explain(), validate_while_typing=False,
                          bottom_toolbar=' Tab completes the folder name, Enter takes it, Ctrl+C to quit ')
        except (KeyboardInterrupt, EOFError):
            raise Stop() from None
        return _clean_path(text)
    while True:
        text = _input(f'     {question} ' + (f'[{default}] ' if default else ''))
        path = _clean_path(text) or _clean_path(default)
        if text.strip() == '-' and allow_empty:
            return ''
        if path and check(path):
            return path
        say(f'     {BAD} ' + (refusal(path) if path else 'Type a folder.'))


def ask_text(question, default=''):
    """A line of text; an empty answer takes the default, shown greyed in the empty field."""
    if rich():
        try:
            text = prompt(FormattedText([('bold', f'     {question} ')]),
                          placeholder=FormattedText([('ansibrightblack', default)]) if default else None)
        except (KeyboardInterrupt, EOFError):
            raise Stop() from None
    else:
        text = _input(f'     {question} ' + (f'[{default}] ' if default else ''))
    return text.strip() or default


def ask_number(question, default, lo, hi, zero=None, suffix=''):
    """A whole number from lo to hi. zero: what 0 means (shown as the default when it is 0)."""
    shown = zero if (zero and default == 0) else f'{default}{suffix}'
    hint_text = f'{lo} to {hi}' + (f', 0 = {zero}' if zero else '')
    while True:
        text = ask_text(f'{question} ({hint_text}) [{shown}]:', '')
        if not text:
            return default
        text = text.rstrip('%').strip()
        if text.isdigit() and lo <= int(text) <= hi:
            return int(text)
        say(f'     {BAD} Type a number from {lo} to {hi}.')


def confirm(question, yes='Yes', no='No'):
    """Yes or no, no by default (the window's dialogs put the action on the left, Cancel on the right)."""
    if rich():
        return choose(question, [(True, yes), (False, no)], default=False)
    return _input(f'     {question} [y/N] ').strip().lower() in ('y', 'yes')


def pause(text='Press Enter to go on.'):
    _input(f'     {text}')


def run_logged(work, quiet_after=20):
    """Runs work(log) with each line it logs stamped with the time so far, and a line now and then
    while a long step (making a track's data) is quiet, so a few minutes does not look like a hang.
    Returns work's result; its exceptions pass through."""
    t0 = time.time()
    last = [t0]
    done = threading.Event()

    def stamp():
        s = int(time.time() - t0)
        return f'{s // 60}:{s % 60:02d}'

    def log(line):
        last[0] = time.time()
        say(f'  [{stamp()}] {line.strip()}')

    def ticker():
        while not done.wait(5):
            if time.time() - last[0] >= quiet_after:
                last[0] = time.time()
                say(f'  [{stamp()}] still working...')

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    try:
        return work(log)
    finally:
        done.set()
        thread.join()
