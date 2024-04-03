def string_to_dict(s):
    return {char: s.count(char) for char in s}

print(string_to_dict("female"))
