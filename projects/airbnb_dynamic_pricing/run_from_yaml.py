import yaml
from tools.jobs.job_base import BasicSeedJob

with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

print(config)

output = BasicSeedJob(**config).run()

print(output["score"])
