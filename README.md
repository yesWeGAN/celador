# celador
Finding beauty in the unexpected. 

17.10.22:
investigate into: https://www.sbert.net/examples/applications/image-search/README.html

Built a library to match Images with their respective embeddings in codebase.*.

Aimed at 100k+ images scraped with Bing Image Downloader, DuckDuckGo or any other mass downloader of your choice. 

Automatic embedding through EfficientNetB0, captioning via OFA-Sys large and embedding the caption via SentenceBert.

Building lookup index via faiss library on image, caption, and combined embedding level.

Enables discovery of samples through large-scale image databases based on visual and semantic similarity.
