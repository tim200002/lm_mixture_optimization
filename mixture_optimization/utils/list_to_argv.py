def list_to_argv(params):
    out = []
    for param in params:
        if isinstance(param, tuple):
            out.append(f"--{param[0]}")
            out.append(str(param[1]))
        else:
            out.append(f"--{param}")
    
    return out


if __name__ == "__main__":
    params = [
        "param1",
        ("param2", 2),
        "param3",
        ("param4", 4),
    ]
    print(list_to_argv(params))  # --param1 --param2 2 --param3 --param4 4