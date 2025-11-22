import pandas as pd

dfs = []
for f in [
    "logs_to_templates_openstack.csv",
    "logs_to_templates_spark.csv",
    "logs_to_templates_zookeeper.csv",
    "logs_to_templates_apache.csv",
    "logs_to_templates_healthapp.csv",
    "logs_to_templates_hdfs.csv",
    "logs_to_templates_linux.csv",

]:
    df = pd.read_csv(f)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined["line_idx"] = range(len(combined))

combined.to_csv("logs_to_templates_combined.csv", index=False)
print("OK → logs_to_templates_combined.csv")
