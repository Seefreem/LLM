When we use load_dataset() for loading a dataset for the first time.
The function will download dataset and save it in the default directory "~/.cache/huggingface/datasets/downloads/".
Then the load_dataset() will generates arrow files of he dataset based on the downloaded compressed files, and save the arrow file in "~/.cache/huggingface/datasets/{dataset_name}/".     
When we delete "~/.cache/huggingface/datasets/{dataset_name}/", then the load_dataset() will generates arrow files again.  
the original data are saved in "~/.cache/huggingface/datasets/downloads/".  
Transforms applied on datasets will be cached to accelerate preprocessing. And this can be a problem, sometimes. Because errors are also saved.   

- Reference reading:
https://huggingface.co/docs/datasets/en/cache  



