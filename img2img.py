import webuiapi

api = webuiapi.WebUIApi()
api.set_auth('username', 'password')

result1 = api.txt2img(
    prompt="cute girl with short brown hair in black t-shirt in animation style",
    negative_prompt="ugly, out of frame",
    seed=-1,
    styles=["anime"],
    cfg_scale=7,
)

modified_image = result1.image
test = modified_image.info
print(test)
modified_image.save("output1.png")

result2 = api.img2img(images=[result1.image], prompt="cute cat", seed=-1, cfg_scale=6.5, denoising_strength=0.6)
modified_image = result2.image
modified_image.save("output.png")

# result3 = api.img2img(images=[result2.image], prompt="cute dog", seed=-1, cfg_scale=6.5, denoising_strength=0.6)
# modified_image = result3.image
# modified_image.save("output2.png")

