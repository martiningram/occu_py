from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

r_times = glob("./speed_test/*/unmarked/time*.csv")
python_times = glob("./speed_test/*/python/*.txt")

py_sizes = [int(x.split("/")[2]) for x in python_times]
r_sizes = [int(x.split("/")[2]) for x in r_times]

loaded_python = [float(list(open(x))[0].strip()) for x in python_times]
loaded_r = [pd.read_csv(x)["x"].iloc[0] for x in r_times]

py_series = pd.Series(loaded_python, index=py_sizes)
r_series = pd.Series(loaded_r, index=r_sizes)

timings = pd.concat(
    [py_series, r_series], axis=1, keys=["occu_py", "unmarked"]
).sort_index()

print("Timings in seconds:")
print(timings)

f, ax = plt.subplots(1, 1)

ax.plot(timings["occu_py"].index, timings["occu_py"] / 60, marker="o", label="Proposed")
ax.plot(
    timings["occu_py"].index, timings["unmarked"] / 60, marker="o", label="Unmarked"
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.grid(which="major")

ax.set_xlabel("Number of checklists")
ax.set_ylabel("Runtime (minutes)")
ax.legend()

f.set_size_inches(8, 4)
f.tight_layout()

plt.savefig("speed_plot.png", dpi=300)
