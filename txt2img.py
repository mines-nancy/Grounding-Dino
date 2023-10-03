import webuiapi

api = webuiapi.WebUIApi()
api.set_auth('username', 'password')

api.util_set_model("AOM3B2_orangemixs.safetensors")

result1 = api.txt2img(
    prompt="teletubies in horror style",
    negative_prompt="",
    seed=-1,
    cfg_scale=7,
    sampler_index="DPM++ 3M SDE Karras",
    width=512,
    height=512
)


modified_image = result1.image
api_info = result1.info


modified_image.save("output.png")

modified_image.show()
