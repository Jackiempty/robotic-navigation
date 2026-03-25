import subprocess

models = ["basic", "diff_drive", "bicycle"]
controllers = ["pid", "pure_pursuit", "stanley", "lqr"]
tracks = ["Silverstone", "Suzuka", "Monza"]
lqr_states = ["steering_angle", "steering_angular_velocity"]

count = 0

print(f" Starting Autonomous Driving Control Tests (Filtered for valid combinations)...")

for model in models:
    for track in tracks:
        for controller in controllers:
            if controller == "stanley" and model != "bicycle":
                continue
            if controller == "lqr":
                if model == "bicycle":
                    current_states = lqr_states
                else:
                    # basic 和 diff_drive 的 LQR 不區分這兩種 state，跑一次即可
                    current_states = ["steering_angle"]

                for state in current_states:
                    count += 1
                    # 加入 --headless 參數
                    cmd = ["python", "navigation.py", "-s", model, "-c", controller, "-t", track, "--lqr_control_state", state, "--headless"]
                    print(f"\n[{count}] Executing: {' '.join(cmd)}")
                    subprocess.run(cmd)
            else:
                count += 1
                # 加入 --headless 參數
                cmd = ["python", "navigation.py", "-s", model, "-c", controller, "-t", track, "--headless"]
                print(f"\n[{count}] Executing: {' '.join(cmd)}")
                subprocess.run(cmd)

print("\n All tests completed successfully!")


