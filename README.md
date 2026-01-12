### create blog 
```bash
cd blog
jupyter nbconvert histogram.ipynb --to markdown --template=custom_markdown.tpl --NbConvertApp.output_files_dir='../img' 
```