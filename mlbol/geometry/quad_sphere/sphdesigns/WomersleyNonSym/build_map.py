import glob
strings = sorted(glob.glob("sf*"))
def convert_and_save(strings, filename):
    with open(filename, 'w') as file:
        for s in strings:
            if s.startswith('sf'):
                parts = s[2:].split('.')
                if len(parts) == 2:
                    a, b = parts
                    file.write(f"{int(b)} {int(a)}\n")
convert_and_save(strings, "map.txt")