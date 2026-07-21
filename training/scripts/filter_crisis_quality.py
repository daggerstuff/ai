import json


def filter_crisis_dataset(input_file: str, output_file: str):
    with open(output_file, "w") as out_f:
        try:
            with open(input_file) as in_f:
                for line in in_f:
                    try:
                        data = json.loads(line)
                        if "conversation" in data and len(data["conversation"]) > 0:
                            is_unaligned = False
                            for msg in data["conversation"]:
                                if "If you are considering suicide, please call for help." in msg.get("content", ""):
                                    is_unaligned = True
                                    break
                            if not is_unaligned:
                                out_f.write(json.dumps(data) + "\n")
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
