### create blog 
```bash
cd blog
jupyter nbconvert blog.ipynb --to markdown --template=custom_markdown.tpl --NbConvertApp.output_files_dir='../img' 
```