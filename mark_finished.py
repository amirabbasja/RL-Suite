import os
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
runs_dir = os.path.join(base_dir, "runs_data")

if not os.path.isdir(runs_dir):
    print(f"runs_data folder not found: {runs_dir}")
else:
    for name in os.listdir(runs_dir):
        folder_path = os.path.join(runs_dir, name)
        if os.path.isdir(folder_path):
            conf_path = os.path.join(folder_path, "conf.json")
            if os.path.isfile(conf_path):
                try:
                    with open(conf_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["finished"] = True
                    data["stop_condition"] = {
                        "maxEpisodes": 99999999999,
                        "maxAvgPoint": 99999999999
                    }
                    with open(conf_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"Updated: {conf_path}")
                except Exception as e:
                    print(f"Skipping {conf_path}: {e}")