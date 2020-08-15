# detectron2-ingest

Video ingestion code that runs over detectron2

The ingestion code is meant to run in a [google colab doc](https://colab.research.google.com/drive/1Zms2mU9tMpZsqvGzqzxMpG5o1Ut1Q4V6?authuser=1#scrollTo=9_FzH13EjseR) 

Other links:

* [Detrectron2 on GitHub](https://github.com/facebookresearch/detectron2)
* [Detectron2 Beginner's Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
* [Detectron2 Model Output Format](https://detectron2.readthedocs.io/tutorials/models.html#model-output-format)
* [Detectron2 instances structure documentation](https://detectron2.readthedocs.io/_modules/detectron2/structures/instances.html)
* [Detectron2 Metadata for Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html#metadata-for-datasets)


## Attempt at running locally (unsuccessful)
Since debugging on google Collab is, shall we say, extremely challenging, I attempted to run and debug locally. My 2013 MBP has a nVidia GPU.

To manage dependencies we use [pipenv](https://docs.python-guide.org/dev/virtualenvs/). This tracks dependencies in [Pipfile](./Pipfile). To install all dependies run:

`$ pipenv install`

*Warning:* be prepared to spend *hours* installing locally…

Furthermore you may encounter timeouts during `pipenv` installations, to correct these try:

`$ export PIPENV_TIMEOUT=9999`

Since MacOS comes with python 2.7.3 by default we also need to upgrade locally to python 3 [following these instructions](https://opensource.com/article/19/5/python-3-default-mac#what-to-do)
