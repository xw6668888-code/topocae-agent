from utils.cae_query_ground.formal_diagnose import diagnose

if __name__ == "__main__":
    import json

    print(json.dumps(diagnose(), ensure_ascii=False, sort_keys=True))

