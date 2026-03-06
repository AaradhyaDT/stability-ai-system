import joblib
import numpy as np
import time

model = joblib.load("model/stability_model.joblib")

print("Starting real-time stability prediction...\n")

try:
    while True:
        tilt_x = np.random.uniform(-10, 10)
        tilt_y = np.random.uniform(-10, 10)
        angular_velocity = np.random.uniform(0, 5)

        data = [[tilt_x, tilt_y, angular_velocity]]

        prediction = model.predict(data)[0]

        status = "STABLE" if prediction == 1 else "UNSTABLE"

        print(f"TiltX:{tilt_x:.2f} TiltY:{tilt_y:.2f} AV:{angular_velocity:.2f} -> {status}")

        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping simulation.")