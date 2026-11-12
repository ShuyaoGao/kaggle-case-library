"""用 CLIP 分词器切分作者的 5 句提示词，存为 tokens.json（网页实验 B 用）。
作者第 19 格用的是 openai/clip-vit-large-patch14 的分词器；SD v1.5 用的是同一套词表，这里直接读 SD v1.5 ONNX 仓库里的 tokenizer。
用法：python -I -X utf8 tokens.py <SD v1.5 模型目录>
"""
import json, os, sys
from transformers import CLIPTokenizer

tk = CLIPTokenizer.from_pretrained(os.path.join(sys.argv[1], "tokenizer"))
P = {
    "cave": "a curious explorer discovers a massive sprawling underground city in a huge cave system. city has churches, european - style buildings and big towers, and is really far away, illuminated by dim ambient lighting. waterfalls are flowing between different levels of the city. award winning digital art, concept art, breathtaking, imaginative, detailed., 8k",
    "girl": "a ultradetailed beautiful panting of a stylish woman with a sword, by conrad roset, greg rutkowski and makoto shinkai, trending on artstation , 8k",
    "synth": "futuristic synthwave city, retro sunset, crystals, spires, volumetric lighting, studio Ghibli style, rendered in unreal engine with clean details --ar 9:16 --hd --q 2",
    "glass": "postapocalyptic city turned to fractal glass, shiny : : close shot : : 3 5 mm, realism, octane render, 8 k, exploration, cinematic, trending on artstation, realistic, 3 5 mm camera, unreal engine, hyper detailed, photo - realistic maximum detail, volumetric light, moody cinematic epic concept art, realistic matte painting, hyper photorealistic, concept art, volumetric light, cinematic epic, octane render, 8 k, corona render, movie concept art, octane render, 8 k, corona render, cinematic, trending on artstation, movie concept art, cinematic composition, ultra - detailed, realistic, hyper - realistic, volumetric lighting, 8 k",
    "anime": "Cybernetic cloaked anime character concept design, dynamic pose, fantasy anime, dark, bejewelled and encrusted technological royal cloak, powerful aggressive sword stance, biological human face, iridescent, dark and intricate, Greg Rutkowski, Makoto Shinkai, anime CGI, animated, animation, artgerm, artstation, digital illustration, 8k",
}
out = {k: [t.replace("</w>", "·") for t in tk.convert_ids_to_tokens(tk(v).input_ids)] for k, v in P.items()}
for k, v in out.items():
    print(k, len(v), "tokens", "(truncated)" if len(v) > 77 else "")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.json"), "w", encoding="utf8"), ensure_ascii=False, separators=(",", ":"))
