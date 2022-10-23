# frame-transformer

This fork is mainly a research fork and will change frequently and there is like zeo focus on being user friendly. I am happy to answer any and all questions however.

The current architecture being trained is in frame_transformer_v4.py. I will be training larger variants of this network after this one so I will be uploading more checkpoints over time; I'm currently using mega.nz since I don't really know anywhere better. The architecture is a 1d u-net in that it uses a stride of (2,1) for downsampling instead of 2. Each encoder and decoder is followed by a frame transformer encoder and frame transformer decoder respectively. Frame transformer modules use the transformer architecture extended into the channel dimension which relies on multichannel linear layers, my implementation of parallel linear layers with an optional depth-wise transform included (so 1x1 conv basically). The frame transformer modules output a specified number of channels that are concatenated to their input and sent down through the u-net. For frame transformer decoders, it uses the u-nets skip connection as memory to allow for frames to attend to the pre-downsampled sequence. It relies on multichannel multihead attention in order to compute attention across each extracted channel in parallel with each channel being partitioned into the number of heads. The qkv projections use a multichannel linear with no depth-wise component followed by a separable 1xN convolution which is an extension of the primer architectures use of conv1d into the channel dimension. I make use of a dataset that I refer to as voxaug to create synthetic data using vocal stem tracks along with instrumental songs; this has proven to be quite effective at teaching the network more nuanced models. 

Due to the use of the transformer architecture I do not pad training items before processing (also helped save space when I had less ssds, was a problem for a while...). Instead, the inferencing scripts take in a padding option which will add padding in two ways. It will pad the start and end of the song with zero padding, and for each batch item it will pad both ends of the input with more context in order to contextualize the audio its working with; in my tests this has proven to be extremely effective in dealing with problematic areas like abrupt vocals or fretless bass. I also include JSON conversion playlists for the inferences script, though I need to clean that stuff up a bit (I'm pretty sure the output directory code in my fork was submitted by someone in a PR for the main repo, so this repo technically is behind the main repo when it comes to that but does include the functionality; if you specify an output directory, the output property in the conversion json file will be put in there).

I am training this on around 31 days of instrumental music and 1839 vocal tracks (idk exactly how many instrumental songs at this point but there are a lot and include many genres, if they were all 4 minutes it would have around 11,147 songs to mix with the 1839 vocal tracks of varying genres for a total of 20,499,639 unique song pairs). There is also a 978 song subset of my dataset that consists of instrumental songs with their vocal counterparts so that the neural network can learn the relationship between vocals and musical cues as that is one piece of information lost when randomly coupling vocals to instrumental music as the voxaug dataset does.

Update on the below checkpoint: still training, currently at epoch 48 but I inspected around 1364 of my vocal tracks (and yes, it was very fun) and found 422 vocal tracks with faint instrumental backings, ranging from extremely audible acoustic guitar to barely audible hihats at quiet moments in a hiphop vocal track. I am allowing the model to train for 89 epochs which will bring it to 750k optimization steps; this should take a few more days. I tested the model at varying cropsizes which shows the model does better with more context: * validation loss (256) = 0.000764, validation loss (512) = 0.000753, validation loss (1024) = 0.000722, validation loss (2048) = 0.000709

Preview checkpoint at the 20th epoch / 174,108 optimization steps (192,699,428 parameters, validation loss of 0.001519 at a cropsize of 256 but still dropping most epochs) for frame_transformer_v4.py and inference_v4.py is here: https://mega.nz/file/SkhhhQTS#PzzTgrroud8yz_TW8agQfImGRWBwdlLwOcHIs4KH1rk (this uses the deafult hyperparameters in train4.py and inference_v4.py). Edit: still training, currently at 35th epoch / 292,252 optimization steps and validation loss is still steadily dropping, albeit somewhat slowly at this point but I guess that's expected later on in training (validatoin loss is currently at 0.001495 but seems to be dropping most epochs still). I'm not really sure at what point I would consider this fully trained; at current pace of training, it will take months to complete one true epoch of my dataset since it has over 20 million combinations of songs which split up into an average of 5 sections of 2048 frames would be over 100 million training examples. Not really sure pretraining is even necessary at this point, not sure what it would really gain if anything considering the size of my dataset...

I also have a pretraining script and around 90+ days worth of general music with and without vocals. I will shift to pretraining once I get a fully trained checkpoint with the standard training script. I also have a denoising diffusion probablistic model implementation of the model, however I'm probably going to write a latent diffusion model from scratch instead in order to gain some speed when sampling.

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", https://arxiv.org/abs/2104.09864
- [9] Asiedu et all., "Decoder Denoising Pretraining for Semantic Segmentation", https://arxiv.org/abs/2205.11423
