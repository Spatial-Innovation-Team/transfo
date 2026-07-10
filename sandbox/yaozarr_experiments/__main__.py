import yaozarrs as yz


# No support for 0.6 yet, so no scene but they do have some notion of transforms

scale = yz.v05.ScaleTransformation(scale=[0.5, 2.0])
translation = yz.v05.TranslationTransformation(translation=[2.0, 3.0])

# there's a v0.6.0 PR here
#https://github.com/imaging-formats/yaozarrs/pull/52
#but it's heavily AI generated, so I won't explore
# it until the author has reviewed it


## they don't actuallt perform the transforms.
# scale.apply(...) # so this doesn't exist






