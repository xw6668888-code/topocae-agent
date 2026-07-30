from utils.cae_query_ground.formal_determinism_resume import verify

if __name__ == "__main__":
    import json

    print(json.dumps(verify(), ensure_ascii=False, sort_keys=True))

