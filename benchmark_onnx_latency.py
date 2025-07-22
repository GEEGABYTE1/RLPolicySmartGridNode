import onnxruntime as ort
import numpy as np
import time
import matplotlib.pyplot as plt
import os


model_path = "actor.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


latent_dim = 16
batch_size = 1
inputs = {session.get_inputs()[0].name: np.random.randn(batch_size, latent_dim).astype(np.float32)}


for _ in range(10):
    session.run(None, inputs)


timings = []
N = 1000
for _ in range(N):
    inputs["latent"] = np.random.randn(batch_size, latent_dim).astype(np.float32)
    start = time.perf_counter()
    _ = session.run(None, inputs)
    end = time.perf_counter()
    timings.append((end - start) * 1000)  

timings = np.array(timings)
mean_latency = np.mean(timings)
std_latency = np.std(timings)

print(f"✅ ONNX inference benchmark complete over {N} runs")
print(f"Average latency: {mean_latency:.3f} ms | Std dev: {std_latency:.3f} ms")


os.makedirs("images", exist_ok=True)
plt.hist(timings, bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Inference Time (ms)")
plt.ylabel("Frequency")
plt.title("ONNX Actor Inference Latency")
plt.grid(True)
plt.savefig("images/onnx_latency_histogram.png")
print("📊 Saved latency histogram to images/onnx_latency_histogram.png")
