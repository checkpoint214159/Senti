import minedojo

env = minedojo.make(
    "open-ended",
    image_size=[640, 1024],
    world_seed="Enter the Nether",
    start_position=dict(x=-260, y=86, z=173, yaw=0, pitch=0),
    drawing_str="""
    <DrawCuboid x1="-262" y1="86" z1="175" x2="-258" y2="86" z2="175" type="obsidian"/>
    <DrawCuboid x1="-262" y1="86" z1="175" x2="-262" y2="91" z2="175" type="obsidian"/>
    <DrawCuboid x1="-258" y1="86" z1="175" x2="-258" y2="91" z2="175" type="obsidian"/>
    <DrawCuboid x1="-262" y1="91" z1="175" x2="-258" y2="91" z2="175" type="obsidian"/>
    <DrawCuboid x1="-200" y1="86" z1="100" x2="-300" y2="86" z2="200" type="diamond_block"/>
    """
)

env.reset()


action = env.action_space.no_op()
env.step(action)

input()
env.close()
