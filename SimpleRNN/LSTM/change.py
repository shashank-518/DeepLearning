import h5py, json

path = r"C:\Users\shash\Documents\DeepLearning\SimpleRNN\LSTM\nextword.h5"
new_path = r"C:\Users\shash\Documents\DeepLearning\SimpleRNN\LSTM\nextword_fixed.h5"

with h5py.File(path, "r") as f:
    model_config = f.attrs.get("model_config")

    if model_config:
        config_str = model_config.encode("utf-8")
        config = json.loads(config_str)

        # Remove all 'time_major' keys recursively
        def remove_time_major(obj):
            if isinstance(obj, dict):
                obj.pop("time_major", None)
                for k in obj:
                    remove_time_major(obj[k])
            elif isinstance(obj, list):
                for item in obj:
                    remove_time_major(item)

        remove_time_major(config)

        # Save fixed config into a new model file
        with h5py.File(new_path, "w") as out_f:
            for key in f.attrs:
                if key == "model_config":
                    out_f.attrs[key] = json.dumps(config).encode("utf-8")
                else:
                    out_f.attrs[key] = f.attrs[key]

            # Copy all datasets and groups (weights, optimizer, etc.)
            f.copy("/", out_f)

        print(f"✅ Fixed model saved at: {new_path}")

    else:
        print("❌ No model_config found — check your .h5 file path")
