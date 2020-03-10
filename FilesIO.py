import jsonpickle
import json


def load_JSON_data(data_json_file):
    with open(data_json_file, "r") as read_file:
        data = jsonpickle.decode(read_file.read())
    return data


def save_JSON_data(data, data_json_file, beautify=False):
    with open(data_json_file, "w") as write_file:
        json_string = jsonpickle.encode(data)
        # beautify - organize the JSON input (adds new lines & indents to make it more readable)
        if beautify:
            parsed_json = json.loads(json_string)
            json_string = json.dumps(parsed_json, indent=4, sort_keys=True)
        write_file.write(json_string)