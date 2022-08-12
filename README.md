# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This architecture is what I'm calling a frame primer. In comparison with the original repo there are many differences, however the most apparent will be the use of a variant of the primer architecture (a transformer variant) as well as a single residual u-net that only downsamples on one dimension. Each encoder and decoder in the residual u-net is followed by either a frame primer encoder or a frame primer decoder. Frame primer modules extract a specified number of channels from the convolutional feature maps and then calculates attention in parallel; I call this form of attention "multichannel multihead attention." These attention maps are then concatenated with the input as the single LSTM channel in MMDENSELSTM and is then sent through to the next layer of the residual u-net. Frame primer decoders, instead of making use of memory from a sequence of encoders, makes use of the skip connection in the residual u-net which includes the attention maps from the corresponding frame primer encoder. All normalization layers also use layer norm; for convolutional blocks this means reshaping to [B,W,H*C] and normalizing H*C. From testing, this architecture appears to work better than the convolutional variant when you have access to large amounts of data which is where the next piece of this repo comes into play. I make use of an augmentation technique that I unoriginally call voxaug, though with image processing this is pretty common I think. I have a large collection of instrumental songs along with 1200+ vocal tracks and have the dataloader randomly mix and match pieces of these tracks to create data on the fly which ensures that all mix tracks will be a perfect sum of instruments + vocals. While this doesn't seem to have a drastic impact on validation loss below, it allows for the model to learn far more nuance and for instance avoid being tripped up by instruments like fretless bass which seem to pose a challenge for most architectures (the below checkpoint with the current inference code typically has no issue with fretless bass at all).

Current training has been paused, I have a teaser checkpoint here: https://mega.nz/file/m45HmLJZ#J8eQqrI1zJcUvX8Imyu8OTF_YeOnjd5FRVOxIMIz91M - this is at the 12th mini epoch, so it would have seen the full dataset at most 3 times (though really far less with augmentations). I did notice a few notes partially removed when listening to music with fretless bass and have since purchased 5 solo bass albums that feature fretless bass to alleviate this issue; I will be continuing training from this checkpoint but resetting the learning rate decay so that it has a chance to see the new material with its highest learning rate. Hyperparameters for this model are: { channels = 16, num_bands=[16, 16, 16, 8, 4, 2], bottlenecks=[1, 2, 4, 8, 12, 14], feedforward_dim = 12288, dropout 0.1, cropsize 512, adam, no amsgrad, no weight decay, max learning rate of 1e-4, num_res_blocks = 1 }.

Still working on comparing this fork to the original repo. This current graph shows four runs: ![image](https://user-images.githubusercontent.com/30326384/183276706-242271c0-b519-4349-9d71-1cbaa10d2589.png)

Run details:

drawn-mountain-926 is a run on my full dataset with the frame primer architecture 118,155,192 parameters FramePrimer2(channels=16, feedforward_dim=12288, n_fft=2048, dropout=0.1, num_res_blocks=1, num_bands=[16, 16, 16, 8, 4, 2], bottlenecks=[1, 2, 4, 8, 12, 14])

pleasant-gorge-925 is tsurumeso's architecture that obviously heavily heavily inspired mine. 129,170,978 parameters - CascadedNet(args.n_fft, 96, 512)

smaller dataset tests:
silvery-butterfly-923 is my architecture on a small subset of the dataset (768 songs I think).

logical-thunder-922 is tsurumeso's architecture on a small subset of the dataset as above

Validation loss is actually quite similar between the two architectures with the frame primer doing better on the full dataset while tsurumeso's architecture does better on a smaller dataset. The validation loss for both architectures is dropping at an equal rate currently on the larger dataset, will need to take training further with both however after 33,243K batches the frame primer is currently doing marginally better than a large version of the cascade net. Given the difference in output between the two architectures, this seems to imply that a combination of the two would produce the best results.

After I finsih training further, I will start on a full pretraining session using the frame primer architecture. Given that it seems to do better with much more data, pretraining should unlock even more potential. I find myself curious about applying this architecture to images as well, as you could simply have vertical and horizontal column attention at that point which would be far cheaper than attention between each individual feature and would allow to extract multiple channels like this. Could simply have two frame (column?) primers after each encoder for a horizontal and vertical attention pass. Might even be useful here...

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
