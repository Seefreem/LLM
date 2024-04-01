Models: 1B, 7B, and 65B
The number of tokens for training a 1B model is 2T (1 epoch).
The number of tokens for training a 1B model is 2.46T (1.23 epoch).

All of our released models have been trained to at least 2T tokens (a single epoch over our training
data), and some have been trained beyond that by starting a second epoch over the data with a
different shuffling order. The impact of repeating this small amount of data should be negligible
according to prior work (Muennighoff et al., 2023).

大模型需要训练几轮？大模型需要完整地看几次训练集？ 1次足以。也是可以多看几次的。







