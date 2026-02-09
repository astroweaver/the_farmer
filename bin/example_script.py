import farmer_local
farmer_local.validate()

# get one brick + run it
brick = farmer_local.build_bricks(1)

# detect sources
brick.detect_sources()
brick.write(allow_update=True)

# # determine models
brick.process_groups(51, mode='model')
brick.write(allow_update=True)

# # force photometry
brick.process_groups(51, mode='photometry')
brick.write(allow_update=True)

group = brick.spawn_group(51)
group.plot_summary()

# write ancillary
brick.build_all_images()
brick.plot_image()
brick.write(allow_update=True)
