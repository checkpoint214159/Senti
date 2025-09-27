from minerl.env.malmo import MalmoEnv

env = MalmoEnv(
    mission_file="test.xml",   # XML with the <WorldPath> tag
    role=0,
    exp_uid="load_existing_world"
)

obs = env.reset()