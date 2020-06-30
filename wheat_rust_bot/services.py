from typing import Any, List
import asyncio
import json
import logging
import os

from albumentations import Compose, LongestMaxSize, Normalize, PadIfNeeded
from albumentations.torch import ToTensor
from common import rpc
import cv2

import torch


@rpc()
def get_shape(*imgs) -> List[Any]:
    return [i.shape for i in imgs]


@rpc()
async def get_square(*vals) -> List[float]:
    await asyncio.sleep(1)
    return [(v * v) for v in vals]


class ClassifyModel:
    def __init__(self):
        self.model = None
        self.class2tag = None
        self.tag2class = None
        self.transform = None

    def load(self, path="/model"):
        image_size = 512
        self.transform = Compose(
            [
                LongestMaxSize(max_size=image_size),
                PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),
                ToTensor(),
            ]
        )
        self.model = torch.jit.load(os.path.join(path, "model.pth"))
        with open(os.path.join(path, "tag2class.json")) as fin:
            self.tag2class = json.load(fin)
            self.class2tag = {v: k for k, v in self.tag2class.items()}
            logging.debug(f"class2tag: {self.class2tag}")

    @rpc(name="classify", pool_size=1, batch_size=4)
    def predict(self, *imgs) -> List[str]:
        logging.debug(f"batch size: {len(imgs)}")
        input_ts = [self.transform(image=img)["image"] for img in imgs]
        input_t = torch.stack(input_ts)
        logging.debug(f"input_t: {input_t.shape}")
        output_ts = self.model(input_t)
        output_ts = torch.softmax(output_ts, dim=1)
        logging.debug(f"output_ts: {output_ts.shape}")
        #logging.debug(f"output_pb: {output_pb}")

        res = []
        for output_t in output_ts:
            tag  = self.class2tag[output_t.argmax().item()]
            prob = output_ts.max()
            res_dict = dict(zip(
                         list(self.tag2class.keys()),list(output_t.numpy())
                       ))
            logging.debug(f"all results: {res_dict}")
            logging.debug(f"prob: {prob}")
            logging.debug(f"result: {tag}")
            res.append((tag,prob,res_dict))
        return res


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    loop.create_task(get_shape.consume())
    loop.create_task(get_square.consume())

    m = ClassifyModel()
    m.load()
    loop.create_task(m.predict.consume())

    loop.run_forever()
