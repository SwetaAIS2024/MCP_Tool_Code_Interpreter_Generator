content = open('test_a2a.py', encoding='utf-8').read()

# Map mojibake sequences to clean ASCII
mapping = {
    '\xe2\x80\x93': '--',     # â€" (en-dash)
    '\xe2\x86\x92': '->',    # â†' (right arrow)
    '\xe2\x94\x80': '-',     # â"€ (box horizontal)
    '\xe2\x80\x94': '--',    # â€" (em-dash)
}

for bad, good in mapping.items():
    content = content.replace(bad, good)

open('test_a2a.py', 'w', encoding='utf-8').write(content)

remaining = [(i+1, line) for i, line in enumerate(content.splitlines())
             if any(ord(c) > 127 for c in line)]
if remaining:
    for lineno, line in remaining:
        print(f'Still garbled line {lineno}: {repr(line[:80])}')
else:
    print('All non-ASCII cleaned. Lines:', content.count('\n'))
