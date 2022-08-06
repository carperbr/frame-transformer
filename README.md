# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

Have changed the architecture once more, new frame primer variant is getting far lower validation loss for first epoch (0.001695 vs the previous best of 0.001726 on my validation set which includes a mix of all kinds of vocals screaming/male singing/female singing); the main module is FramePrimer2 in frame_primer/frame_primer.py. The new architecture is a modification of the original BaseNet from the original repo using the frame primer modules and constraining downsampling only to the frequency dimension. In the new variant, each encoder and decoder are followed by frame primer encoders and decoders respectively. The key change with this new variant is that the attention modules now have a variable bottleneck; this gives the modules the ability to extract multiple representations from the convolutional feature maps and perform attention across the extracted maps using batched matrix multiplication (same concept as multihead attention but I include a 5th dimension for a shape of (batch, channels, num heads, sequence length, head features)). I have also included rotary positional embeddings from https://github.com/lucidrains/rotary-embedding-torch, will likely take a swing at a custom implementation of this as well to get a more intuitive grasp on it, so this should be able to extend to really any sequence length which should hopefully make pretraining less absurd on my dataset.

Current training has been paused, I have a teaser checkpoint here: https://mega.nz/file/m45HmLJZ#J8eQqrI1zJcUvX8Imyu8OTF_YeOnjd5FRVOxIMIz91M - this is at the 12th mini epoch, so it would have seen the full dataset at most 3 times (though really far less with augmentations). I did notice a few notes partially removed when listening to music with fretless bass and have since purchased 5 solo bass albums that feature fretless bass to alleviate this issue; I will be continuing training from this checkpoint but resetting the learning rate decay so that it has a chance to see the new material with its highest learning rate. Hyperparameters for this model are: { channels = 16, num_bands=[16, 16, 16, 8, 4, 2], bottlenecks=[1, 2, 4, 8, 12, 14], feedforward_dim = 12288, dropout 0.1, cropsize 512, adam, no amsgrad, no weight decay, max learning rate of 1e-4, num_res_blocks = 1 }

Still working on comparing this fork to the original repo. This current graph shows four runs: ![image](https://user-images.githubusercontent.com/30326384/183269188-31100335-4627-4d6d-b1a6-ff0864a60aca.png)

Run details:
drawn-mountain-926 is a run on my full dataset with the frame primer architecture 118,155,192 parameters FramePrimer2(channels=16, feedforward_dim=12288, n_fft=2048, dropout=0.1, num_res_blocks=1, num_bands=[16, 16, 16, 8, 4, 2], bottlenecks=[1, 2, 4, 8, 12, 14])

pleasant-gorge-925 is tsurumeso's architecture that obviously heavily heavily inspired mine. 129,170,978 parameters - CascadedNet(args.n_fft, 96, 512)

smaller dataset tests:
silvery-butterfly-923 is my architecture on a small subset of the dataset (768 songs I think).

logical-thunder-922 is tsurumeso's architecture on a small subset of the dataset as above

Validation loss is actually quite similar between the two architectures with the frame primer doing better on the full dataset while tsurumeso's architecture does better on a smaller dataset (0.001655 for tsurumeso's vs 0.001638 for mine). That being said, I need to train my architecture out to another epoch to compare the drop in validation loss between the two and compare the rate of convergence. Next test I want to carry out is going to be a similar setup to tsurumeso's architecture with a frame primer encoder and frame primer decoder following encoder 2 and decoder 2 respectively.

Current training session seems to be going quite well with the FramePrimer2 architecture; should have a fully trained model within a day or two I'd estimate, though not entirely as the drop in validation loss appears to be accelerating even after 42 hours of training... ![image](https://user-images.githubusercontent.com/30326384/182856349-5a4c755e-f68f-4843-b746-144aba32f529.png)

I think my experiments with this architecture are coming to a close, only have one final test I'd like to carry out though I may hold off until I train the current version further. The current frame primer architecture found in frame_primer/frame_primer.py seems to be the most useful architecture I've converged on, though I intend on carrying out one final test with the evolved transformer architecture. I have tried many variants, however it seems that a 6 layer u-net with one encoder & decoder per layer for a total of 12 transformer modules is the ideal setup for this architecture. A frame primer without the u-net portion works somewhat well but takes a significant amount of time to train; column stride mixed with a conv u-net leads to not very great performance, and a normal u-net performs quite a bit worse than u-nets that downsample only on the frequency dimension (after 5.5 hours of training with a model similar to the original repo with a dense net of u-nets minus aux outputs it reached 0.001903 on my validation set vs the frame primer reaching 0.001726 in 4 hours). Due to this, the architecture I have converged on use 1 encoder/decoder per layer, 16 heads of attention, and a feedforward dimension of 12288 (12x 1024 as per the primer paper, athough this increases far past 12x as the input descends into the u-net and is downsampled), one res block per encoder/decoder, and 16 channels per layer (so each layer of the u-net adds 16 channels for a total of 96 channels at its widest section). The architecture only downsamples the frequency axis unlike a typical u-net which enables the use of a residual attention connection as seen in the realformer paper which also seems to help. My current version uses column stride (Nx1) but no longer uses column kernels and now is using 3x3 kernels instead which helps a bit although not very drastically. I also currently map the input to a range of [-1,1]; a short test showed validation loss drop with input within this range vs [0,1] however it was not significant; my main reasoning for doing this is for the pretraining stage so that I can have it predict the noise rather than the output spectrogram as described in "Decoder Denoising Pretraining for Semantic Segmentation."

I am currently shifting into the training phase. The architecture itself seems to be quite accurate and tends to not have volume drop offs I've encountered with other architectures, however I also have an immense dataset I've constructed over the course of about a year. My personal dataset consists of a few parts: an instrumental library, a vocal library, a mix/instrumental pair library, and an unlabeled library. The voxaug dataloader will construct new training data on the fly utilizing the instrumental library (28.58 days of instrumental music, idk how many songs but if they were all 4 minutes it would be 10,290 or so songs) and the vocal library (1,297 vocal stems). This architecture is like other transformers where large amounts of data leads to far better results which was the main reasoning behind coming up with this dataloader. To augment this approximately 13,346,130 song collection I have a collection of mix/instrumental pairs as well (also because rhythmic information can be important to vocal removing). The instrumental/mix dataset consists of 765 songs. The unlabeled dataset consists of around 55.21 days of music now which is around 19,874 additional 4 minute songs not found in the instrumental portion.

My first goal is to get a model trained with purely supervised training uploaded within a few days trained on a cropsize of 512. After this, I will begin pretraining a model using the denoising pretraining discussed in "Decoder Denoising Pretraining for Semantic Segmentation," however I will train the full network with this system and afterward will try finetuning on just the instrumental/mix portion of my dataset (voxaug will hopefully be unneeded at that point given pretraining having immense amounts of data).

I will be making a diagram of the model as well as making a youtube video walking through my architecture. I am under no illusion that anyone is paying attention to this repo, however on the off chance someone sees this and wants to help feel free to reach out to me by leaving a comment on one of my commits or something. My plan is to begin work on a frontend and backend once the model is trained (will aim for both cloud and local inference, though cloud will not be usable without your own cluster as I am not rich and can't afford to host a service like that for free lol)

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Huang et al., "Music Transformer: Generating Music with Long-Term Structure", https://arxiv.org/pdf/1809.04281.pdf
- [8] He et al., "RealFormer: Transformer Likes Residual Attention", https://arxiv.org/pdf/2012.11747.pdf
- [9] Asiedu et all., "Decoder Denoising Pretraining for Semantic Segmentation", https://arxiv.org/abs/2205.11423
