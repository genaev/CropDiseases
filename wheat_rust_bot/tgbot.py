import os

from aiotg import Bot, Chat
import imageio
import json
from services import ClassifyModel, get_square

model = ClassifyModel()

bot = Bot(api_token=os.getenv("TG_TOKEN"))


@bot.command("/start")
async def start(chat: Chat, match):
    return chat.reply("Send me photo of wheat.")


@bot.handle("photo")
async def handle_photo(chat: Chat, photos):
    # Get image binary data
    meta = await bot.get_file(photos[-1]["file_id"])
    resp = await bot.download_file(meta["file_path"])
    data = await resp.read()

    # Convert binary data to numpy.ndarray image
    image = imageio.imread(data)

    # Do the magic
    (tag,prob,res_dict) = await model.predict.call(image)
    if prob > 0.8:
      # Simple text response
      await chat.reply(f"I think this is {tag} (confidence={prob:.2f})")
    else:
      await chat.reply(f"I'm not sure about this image (confidence={res_dict})")

    filename = meta['file_unique_id']
    imageio.imwrite('/imgs/'+filename+'.jpg', image)
    log_data = {'msg': chat.message, 'tag': tag, 'prob': prob.numpy().tolist()}
    with open('/imgs/'+filename+'.json', 'w') as fp:
      json.dump(log_data, fp, indent=2)
    
    # Or image response
    #with open(f"{tag}.jpg", "rb") as f:
    #    await chat.send_photo(f, caption=f"... the {tag} like this one!")


@bot.command(r"/square (.+)")
async def square_command(chat: Chat, match):
    val = match.group(1)
    try:
        val = float(val)
        square = await get_square.call(val)
        resp = f"Square for {val} is {square}"
    except Exception():
        resp = "Invalid number"
    return chat.reply(resp)


bot.run()
