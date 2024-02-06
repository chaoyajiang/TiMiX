***TiMix: Text-aware Image Mixing for Effective Vision-Language Pre-training***


## Pre-trained models and datasets

* Pre-trained models

 
For VQA and image captioning tasks, we do an additional continue pre-training on 4M image-text pairs based mplug.en.large to get mplug.en.large.v2.
 
 
|Model | Visual Backbone | Text Enc Layers | Fusion Layers | Text Dec Layers | #params | Download |
|------------------------|-------------------------------------------|------|------|------|------|-----|

* Pre-train Datasets

                                                                        
| | COCO | VG | SBU | CC3M | CC13M |
|------------------------|-------------------------------------------|------|------|------|------|
|image | 113K | 100K | 860K | 3M | 10M | 
|text | 567K | 769K | 860K | 3M | 10M |





## Requirements
* [PyTorch](https://pytorch.org/) version >= 1.11.0

* Install other libraries via
```
pip install -r requirements.txt
```


## Pre-training


Comming soon.


## Fine-tuning

[Download json files of downstream tasks](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/data.tar)

### Visual Question Answering

1. Download VQA v2 dataset and Visual Genome dataset from the original websites [VQA 2.0](https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278).
2. Download and extract the provided dataset json files.
3. In configs/vqa_mplug_base.yaml, set the paths for the json files and the image paths.
4. Finetune the pre-trained mplug_base or large model using 8 A100 GPUs:
<pre>sh scripts/vqa_mplug_base.sh</pre> 
                                              
5. Evaluate the result using the official evaluation server.
       
                                                                                          
### Image Captioning
     
                                                                                          
1. Download COCO Caption dataset from the original websites.
2. Download and extract the provided dataset json files.
3. Download language evalution tool([language_evalution](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/language_evalution.tar)).
4. In configs/caption_mplug_base.yaml, set the paths for the json files and the image paths.
5. Finetune the pre-trained mplug_base or large model using 8 A100 GPUs:
<pre>sh scripts/caption_mplug_base.sh</pre> 
 

                                                                                          
### Image-text Retrieval
1. Download MSCOCO or Flickr30k datasets from the original websites.
2. Download and extract the provided dataset json files.
3. In configs/retrieval_flickr30k_mplug_large.yaml or configs/retrieval_coco_mplug_large.yaml, set the paths for the json files and the image path.
4. Finetune the pre-trained checkpoint using 8 A100 GPUs:
<pre>sh scripts/retrieval_flickr30k_mplug_base.sh</pre> 
<pre>sh scripts/retrieval_coco_mplug_base.sh</pre>

### Visual Grounding
1. Download RefCOCO datasets from the original websites.
2. Download and extract the provided dataset json files.
3. In configs/grounding_mplug_large.yaml, set the paths for the json files and the image path. Data preparation can follow [TransVG](https://github.com/djiajunustc/TransVG)
4. Finetune the pre-trained checkpoint using 8 A100 GPUs:
<pre> sh scripts/grounding_mplug_base.sh </pre> 


