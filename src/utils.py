import json
import pickle
import yaml
import os 

###############################################
# Helper functions for reading and writing files
###############################################

def dump_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_str(s, path):
    with open(path, "w") as f:
        f.write(s)


def load_str(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return "\n".join(f.readlines())


def handle_non_serializable(obj):
    return "non-serializable contents"  # mark the non-serializable part


def load_json(file_name, encoding="utf-8"):
    with open(file_name, "r", encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, file_name, ensure_ascii=True, indent=4):
    with open(file_name, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def pretify_json(path):
    data = load_json(path)
    output_path = path.replace(".json", "_pretty.json")
    dump_json(data, output_path)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def write_html(obj, path):
    with open(path, "w") as f:
        f.write(obj)


###############################################
# Helper function to print args from argparse
###############################################
import shlex
import sys


def format_args(args, style="half-line"):
    # script_name = os.path.basename(sys.argv[0])
    # script_name = sys.argv[0]
    script_path = os.path.abspath(sys.argv[0])
    cmd = [f"python {script_path}"]

    for key, value in vars(args).items():
        if isinstance(value, bool):
            # Only include arg flags if they are True
            if value:
                # cmd.append(f"--{key.replace('_', '-')}")
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key} {shlex.quote(str(value))}")
            # cmd.append(f"--{key.replace('_', '-')} {shlex.quote(str(value))}")

    separator = " \\\n    "

    if style == "half-line":
        msg = "\n+ ---- ARGS\n"
        msg += separator.join(cmd)
        msg += "\n+ -----\n"
        
    elif style == "box":
        border = "+-------------------------------------------+\n"
        title = "|                 ARGUMENTS                 |\n"
        msg = "\n" + border + title + border
        msg += separator.join(cmd) + "\n"
        msg += border

    return msg
