# 28.12.23
This codebase gets a revival. Update. New purpose.

Goal: Pretrain an index using FAISS on ImageNet, COCO, etc. Anything where captions and image exist.
With that index, then project each new image from a user into a pre-defined index. That one will order your pictures into a graph.
Pictures need to be uploaded to a cloud anyway, so processing can be on heavy machines (don't need to downscale anything).

With that tool, you'd have easy access to your data. You could ask questions (pre-embedded questions for very fast inference), or specific questions,
have faces and names connected, so you can ask for specific events, etc. 
And yes, once implemented, I could use it on 2M datapoints.

Helps you keep organized! Everything is in one place. 

Generally speaking, choose a SPECIFIC problem and offer a SPECIFIC solution (own dataset or other).
Don't reinved the foundation models, just use them for your purposes.

Writeup: 
1. OFA-Sys huge download while it's still there.
2. LLAMA-13b download while it's still there.
3. ViT encoder embeddings
4. https://arxiv.org/abs/2210.10620

Build it. It'll be good. 


# celador
Finding beauty in the unexpected. 

17.10.22:
investigate into: https://www.sbert.net/examples/applications/image-search/README.html

Built a library to match Images with their respective embeddings in codebase.*.

Aimed at 100k+ images scraped with Bing Image Downloader, DuckDuckGo or any other mass downloader of your choice. 

Automatic embedding through EfficientNetB0, captioning via OFA-Sys large and embedding the caption via SentenceBert.

Building lookup index via faiss library on image, caption, and combined embedding level.

Enables discovery of samples through large-scale image databases based on visual and semantic similarity.
