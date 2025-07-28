# FrameTransformer

Been a while, but there will be some large updates here very soon that may get put into a new repository. I've redesigned the neural network architecture using an dual attention-based cross-layer memory mechanism as well as adjusting how it processes spectrograms which has greatly increased accuracy.

I have updated the project with a new neural network inspired by hierarchical transformers and my older frame transformer

The new architecture is a temporal u-net frame transformer rather than a frequency u-net. This has the added benefit of condensing the sequence length akin to hierarchical transformers which allows for more processing power in the transformer. This appears to capture temporal dynamics quite well as well, and with an updated data synthesis technique it appears the quality has surpassed the frequency u-net despite only being at 280k optimization steps thus far. V9 was a bit of a failed experiment though it did teach me some new stuff, specifically that the data synthesis changes were indeed beneficial and that the patch transformer setup isn't very great at capturing temporal dynamics in audio given that you are breaking apart the true temporal dynamics in that process. V10 will continue training until 2 million optimization steps, so it has quite a bit to go until fully trained which is a very promising sign given that it surpassed v9 when v9 was at 1.5mil optimization steps while it was only at 280k optimization steps.

There is a new inferencing script that allows you to mix and match the various models in a min-ensemble method, although I will be doing more with this in the future. I will be using a mutation transformer to create a neural ensemble approach before long, but for now the min method works well.

You can find the new temporal u-net transformer here https://github.com/carperbr/frame-transformer/blob/master/inference/v10/libft2gan/frame_transformer13.py

The YT channel was terminated due to a coordinated barrage of copyright strikes from Sony. Discord community is located here: https://discord.gg/8V5t9ZXRqS

Installation tips from the community with thanks to HunterThompson, PsychoticFrog, mesk and others for helping converge on these, will clean up requirements.txt soon:
https://discord.com/channels/1143212618006405170/1143212618492936292/1148050453645508658

TypeError: init() got an unexpected keyword argument 'dtype'
pytorch is too old, update to 2.0.0+cu118

ModuleNotFoundError: No module named 'einops'
pip install einops

module 'numpy' has no attribute 'complex'.
pip install numpy==1.23.5

librosa 0.8.1 works
librosa 0.9.2 works, but gave me warnings

edit inference.py like this for model version in filenames
https://cdn.discordapp.com/attachments/531122660244062236/1157139846947672115/image.png?ex=651ad1b6&is=65198036&hm=f529e7dcbcb268afa02f41937486ca86e9fa85cd4d0c571cf6e0b0b7d294466e&

weird python issues? Uninstall python, reinstall python and reboot. This wipes all your pip modules so you have to install pytorch, numpy 1.23.5 and einops again

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", https://arxiv.org/abs/2104.09864
- [8] He et al., "RealFormer: Transformer Likes Residual Attention", https://arxiv.org/abs/2012.11747
- [9] Nawrot et al., "Hierarchical Transformers Are More EFficient Language Models", https://arxiv.org/abs/2110.13711
