from argparse import ArgumentParser

arg = ArgumentParser()


def greet(name: str) -> str:
    return f"Hello, {name}!"


if __name__ == "__main__":
    arg.add_argument("--name", type=str, default="World", help="Name to greet")
    args = arg.parse_args()
    print(greet(args.name))
