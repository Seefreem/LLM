# Model configure 
在修改config 文件/对象时，遇到个问题，我想要让模型返回字典，而不是元组。于是想到在模型的配置文件和配置文件类里面加 
use_return_dict = True
但是尝试很久才发现。配置文件类是LlamaConfig， 并且类中并没有 use_return_dict 这个属性。
所以直接修改配置文件的方法肯定是不行的。得直接修改类。或者查找其提供的方法。
但是注意到一个事实，那就是 use_return_dict 实际上是Huggingface引入的。不是LlaMa自带的。

进一步深入了解到，在Evaluation的时候，其实模型返回的是字典。只是在调用自定义的 compute_matric的时候，
输入参数被包装成 EvalPredictionl了。 并且这个类型就两个或者三个参数，
predictions、label_ids和/或inputs

# EvalPrediction.predictions
What does EvalPrediction.predictions contain exactly?
https://discuss.huggingface.co/t/what-does-evalprediction-predictions-contain-exactly/1691
The Trainer will put in predictions everything your model returns (apart from the loss). 
So if you get multiple arrays, it’s likely because your model returns multiple things. 
No one can help you determine what they are without seeing your model 
(which is why you should always post the code you’re using when asking for help :wink: )

outputs['labels'] = outputs['input_ids'] # ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`labels` in this case) have excessive nesting (inputs type `list` where type `int` is expected).
return outputs
# Runtime Errors
## Errors while running load_dataset()
1. OSError: Invalid flatbuffers message. -> Download data again, OR Only load a small amount of data, OR reboot
2. ArrowInvalid: Old metadata version not supported
3. DatasetGenerationError: An error occurred while generating the dataset 
4. OSError: Corrupt snappy compressed data. -> Download data again

## Errors while running Dataset.train_test_split()
1. OSError: Invalid flatbuffers message. -> restart the kernel.

## Errors while running Datasets.map()
1. batch_size=1000: index out of bounds: the len is 31172 but the index is 8589960764 -> decrease the batch size.
2. batch_size=500:  index out of bounds: the len is 30153 but the index is 283467863127
3. ArrowInvalid: Column 1 named input_ids expected length 500 but got length 8
4. IndentationError: unindent does not match any outer indentation level -> Restart the kernel.
5. RuntimeError: One of the subprocesses has abruptly died during map operation.To debug the error, disable multiprocessing. -> Turn off multi-thread processing.
If the map() function crushed, there is a high possibility to fail in later data-loading processes.


