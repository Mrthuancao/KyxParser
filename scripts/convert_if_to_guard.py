#!/usr/bin/env python3
"""
Convert `if (cond) { ... }` and `if (cond) { ... } else { ... }` in .kyx/.txt files
into guarded command form used in the repo, for example:

if (C) { S; }
->
{ ? (C); S; }

if (C) { S; } else { T; }
->
{ ? (C); S; } ++ { ? (!(C)); T; }

Usage: python scripts/convert_if_to_guard.py path/to/file.kyx [more files or glob]
Backups: original files are saved with a .bak extension.

This script uses a small parser to match parentheses and braces and preserves
indentation and inner content.
"""
import argparse
import glob
import os
import sys


def find_matching(text, start, open_ch, close_ch):
    """Find index of matching close_ch for opening at start (text[start]==open_ch).
    Returns index of closing character or -1.
    Handles nested pairs and ignores nothing (no string/comment handling).
    """
    assert text[start] == open_ch
    depth = 0
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def replace_if_blocks(text):
    i = 0
    out_parts = []
    n = len(text)
    import re
    if_pat = re.compile(r"\bif\b")
    while i < n:
        m = if_pat.search(text, i)
        if not m:
            out_parts.append(text[i:])
            break
        start = m.start()
        # append content before 'if'
        out_parts.append(text[i:start])
        # find line indentation
        line_start = text.rfind('\n', 0, start) + 1
        indent = text[line_start:start]
        # find '('
        j = m.end()
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != '(':
            # not a normal if(...), copy 'if' and continue
            out_parts.append('if')
            i = m.end()
            continue
        cond_open = j
        cond_close = find_matching(text, cond_open, '(', ')')
        if cond_close == -1:
            # unmatched, bail out and copy rest
            out_parts.append(text[start:])
            break
        cond = text[cond_open+1:cond_close]
        k = cond_close + 1
        # skip spaces
        while k < n and text[k].isspace():
            k += 1
        # expect then block starting with '{'
        if k >= n or text[k] != '{':
            # not a block we can handle; copy original 'if' and continue
            out_parts.append(text[start:cond_close+1])
            i = cond_close+1
            continue
        then_open = k
        then_close = find_matching(text, then_open, '{', '}')
        if then_close == -1:
            out_parts.append(text[start:])
            break
        then_body = text[then_open+1:then_close]
        L = then_close+1
        # skip whitespace/comments to detect optional else
        while L < n and text[L].isspace():
            L += 1
        has_else = False
        else_body = ''
        if text.startswith('else', L):
            melse_end = L + 4
            # skip spaces
            while melse_end < n and text[melse_end].isspace():
                melse_end += 1
            if melse_end < n and text[melse_end] == '{':
                else_open = melse_end
                else_close = find_matching(text, else_open, '{', '}')
                if else_close == -1:
                    out_parts.append(text[start:])
                    break
                else_body = text[else_open+1:else_close]
                has_else = True
                next_i = else_close + 1
            else:
                # else without block - not handled, copy original
                out_parts.append(text[start:then_close+1])
                i = then_close+1
                continue
        else:
            next_i = then_close + 1
        # build replacement
        cond_str = cond.strip()
        then_str = then_body.strip()
        # preserve inner formatting by keeping body as-is but trimming outer newlines
        if has_else:
            else_str = else_body.strip()
            # use negated condition !(cond)
            repl = indent + "{ ? (" + cond_str + "); " + then_str + " } ++ { ? (!(" + cond_str + ")); " + else_str + " }"
        else:
            repl = indent + "{ ? (" + cond_str + "); " + then_str + " }"
        out_parts.append(repl)
        i = next_i
    return ''.join(out_parts)


def process_file(path, inplace=True):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    new_text = replace_if_blocks(text)
    if new_text == text:
        print(f'No changes for {path}')
        return False
    bak = path + '.bak'
    if not os.path.exists(bak):
        os.rename(path, bak)
    else:
        # if backup exists, overwrite original from current content
        pass
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    print(f'Updated {path} (backup: {bak})')
    return True


def main():
    ap = argparse.ArgumentParser(description='Convert if-else blocks into guarded choices')
    ap.add_argument('paths', nargs='+', help='files or glob patterns to process')
    args = ap.parse_args()
    files = []
    for p in args.paths:
        if any(ch in p for ch in ['*', '?', '[']):
            files.extend(glob.glob(p, recursive=True))
        else:
            files.append(p)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print('No files found')
        sys.exit(1)
    for f in files:
        process_file(f)

if __name__ == '__main__':
    main()
