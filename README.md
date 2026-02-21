# 3dMaskDeepJSCC
Wouldn't Masked Autoencoder structure benefit DeepJSCC in terms of low cbr?
Let's try it out.

# Models
## Channel Mask
1. Using Conv3d to kinda tokenize it in 3d and guess it out. -> WaveNetInspired
2. Using grouped ResNet to Kinda guess it out. -> GroupConv
3. Using Conv3d to tokenize feature vector, Then use transformer to figure out Missing Chunks -> MaskChannel

## Pixel Mask
1. Using Original DeepJSCC as autoencoders, Use MaskGIT style decoding -> On it
2. Is it possible to mask pixels per channel, and restore it parellely? Let's find out. -> Is it possible by extensive parallelism? I think it will work, if I manage to just create MaskGIT.

## Using channel, spatial mask on DeepJSCC
1. Inspired by ChA-MAEViT. 
2. This needs to spread it's receptive field into 3 directions: H, W, C
   How to make this work? I dunno. Maybe extensive engineering on positional embedding? Or the transformer itself needs to be changed completely.
   Maybe architecture from VMamba could help this.
