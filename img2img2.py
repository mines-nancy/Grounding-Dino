import webuiapi
from PIL import Image, ImageDraw

api = webuiapi.WebUIApi()
api.set_auth('username', 'password')

result1 = Image.open("squirtle.jpg")

result2 = api.img2img(images=[result1], denoising_strength=0.0)
modified_image = result2.image

mask = Image.new('RGB', result2.image.size, color = 'black')
# mask = result2.image.copy()
draw = ImageDraw.Draw(mask)
draw.ellipse((130,40,370,340), fill='white')

mask.save("test.png")

result3 = api.img2img(images=[result2.image], mask_image=mask, inpainting_fill=1, sampler_index="Restart", prompt="turtle", seed=-1, cfg_scale=6.5, denoising_strength=0.6)
modified_image = result3.image
modified_image.save("output1.png")

# result3 = api.img2img(images=[result2.image], prompt="cute dog", seed=-1, cfg_scale=6.5, denoising_strength=0.6)
# modified_image = result3.image
# modified_image.save("output2.png")
