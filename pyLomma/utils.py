import hashlib
import os
from timeit import default_timer as timer
from urllib.parse import urlparse

import requests


class benchmark:
    '''Util to benchmark code blocks.
    '''

    def __init__(self, msg, fmt='%0.3g'):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(('%s: ' + self.fmt + ' seconds') % (self.msg, t))
        self.time = t


def as_filename(folder: str, url: str) -> str:
    return os.path.join(folder, os.path.basename(urlparse(url).path))


def assign(value: str, subst: dict[str, str] = None, ch: str = None) -> str:
    assert ch is None or is_variable(ch)
    count = 2 + sum(1 for x in (subst or {}).values() if is_variable(x))
    marker = ch or f'A{count}'
    if '::' not in value:
        return marker

    _, kind = value.split('::', maxsplit=1)
    return f'{marker}::{kind}'


def download(folder: str, url: str, digest: str) -> None:
    filename = os.path.basename(urlparse(url).path)
    fullname = os.path.join(folder, filename)
    if not os.path.exists(fullname) or md5(fullname) != digest:
        with open(fullname, 'wb') as f:
            with requests.get(url) as r:
                f.write(r.content)
        print(f"* Got '{filename}' ({md5(fullname)})")


def generalize(path: tuple[tuple[str, str, str], ...], subst: dict[str, str]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (subst.setdefault(s, assign(s, subst)), subst.setdefault(t, assign(t, subst)), k)
        for s, t, k in path
    )


def md5(filename: str) -> str:
    with open(filename, 'rb') as f:
        return hashlib.file_digest(f, 'md5').hexdigest()


def is_variable(value: str) -> bool:
    if '::' in value:
        value, _ = value.split('::', maxsplit=1)

    return value[0].isupper() or value[0] == '_'


def prettify(atoms: tuple[tuple[str, str, str], ...]) -> str:
    assert len(atoms) >= 2

    atoms = [f'{k}({s},{t})' for s, t, k in atoms]

    return f'{atoms[0]} :- {', '.join(atoms[1:])}.'
