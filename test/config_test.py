import yaml

count = int(0)
config = yaml.safe_load(open("config.yaml", "r"))
for i in config:
    for j in config[i]:
        if config[i][j]:
            count = count + 1
if count == 5:
    print("config is fine")
else:
    print("config is not okay..")
    print("give required file names through config file or in command line..")
